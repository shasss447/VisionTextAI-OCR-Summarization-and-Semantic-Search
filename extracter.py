import pytesseract
import easyocr
import cv2
import numpy as np
import spacy
import re
import os
import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class TesseractProcessor:
    def __init__(self, output_dir: str = "tes_extracted_content"):
        """
        Initialize the document processor for images.
        
        Args:
            output_dir (str): Directory to save extracted images
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        # Set Tesseract path - modify this according to your installation
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        # initializing model and tokenizer
        self.model_name = "facebook/bart-large-cnn" 
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        # Load SpaCy model for NLP tasks
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        # Create output directories
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        
    def process_document(self, image_path: str) -> Dict[str, Any]:
        """
        Process a document image to extract text and images.
        
        Args:
            image_path (str): Path to the document image
            
        Returns:
            Dict[str, Any]: Extracted content including text, entities, and image paths
        """
        try:
            start_time = time.time()
            # Read and process image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
            
            # Extract text content
            text_result = self.extract_text(image)
            
            # Extract images
            image_result = self.extract_images(image, Path(image_path).stem)
            processing_time = time.time() - start_time
            return {
                **text_result,
                'extracted_images': image_result,
                'processing_time': processing_time
            }
            
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    def extract_text(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract and process text from image.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Dict[str, Any]: Processed text data
        """
        # Preprocess image for better OCR
        processed_image = self.preprocess_for_ocr(image)
        
        # Perform OCR
        raw_text = pytesseract.image_to_string(processed_image)
        
        # Get OCR confidence data
        ocr_data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
        
        # Process extracted text
        processed_data = self.process_text(raw_text)
        summary=self.summary_generator(processed_data['processed_text'])
        
        return {
            'raw_text': raw_text,
            'processed_text': processed_data['processed_text'],
            'summary':summary,
            'entities': processed_data['entities'],
            'pos_tags': processed_data['pos_tags'],
            'confidence_scores': self._get_confidence_scores(ocr_data)
        }

    def extract_images(self, image: np.ndarray, base_name: str) -> List[Dict[str, Any]]:
        """
        Extract images from document using contour detection.
        
        Args:
            image (np.ndarray): Input image
            base_name (str): Base name for saving extracted images
            
        Returns:
            List[Dict[str, Any]]: List of extracted image information
        """
        extracted = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply different thresholding methods to detect various types of images
        binary_methods = [
            ('otsu', cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),
            # ('adaptive', cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            #                                 cv2.THRESH_BINARY_INV, 11, 2))
        ]
        
        for method_name, binary in binary_methods:
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for idx, contour in enumerate(contours):
                # Filter small contours and check aspect ratio
                if not self._is_valid_image_region(contour, image.shape):
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract region
                roi = image[y:y+h, x:x+w]
                
                # Check if region contains enough variation to be an image
                if not self._is_image_content(roi):
                    continue
                
                # Save extracted region
                image_filename = f"{base_name}_region_{idx + 1}.png"
                image_path = os.path.join(self.images_dir, image_filename)
                
                cv2.imwrite(image_path, roi)
                
                extracted.append({
                    'filename': image_filename,
                    'path': image_path,
                    'position': {'x': x, 'y': y, 'width': w, 'height': h},
                    'area': cv2.contourArea(contour),
                })
        
        return extracted

    def _is_valid_image_region(self, contour: np.ndarray, image_shape: Tuple) -> bool:
        """
        Check if contour is likely to be an image region.
        """
        # Get area and dimensions
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h != 0 else 0
        
        # Calculate relative size
        image_area = image_shape[0] * image_shape[1]
        relative_size = area / image_area
        
        # Define thresholds
        MIN_RELATIVE_SIZE = 0.01  # 1% of image
        MAX_RELATIVE_SIZE = 0.9   # 90% of image
        MIN_ASPECT_RATIO = 0.2
        MAX_ASPECT_RATIO = 5.0
        
        return (MIN_RELATIVE_SIZE <= relative_size <= MAX_RELATIVE_SIZE and 
                MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO)

    def _is_image_content(self, region: np.ndarray, std_threshold: float = 20) -> bool:
        """
        Check if region contains enough variation to be an image.
        """
        gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        return np.std(gray_region) > std_threshold

    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)
        
        # Deskew if needed
        angle = self._get_skew_angle(denoised)
        if abs(angle) > 0.5:
            denoised = self._rotate_image(denoised, angle)
        
        return denoised

    def _get_skew_angle(self, image: np.ndarray) -> float:
        """Calculate skew angle of the image."""
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        return -angle

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate the image by given angle."""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process extracted text using NLP techniques.
        """
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Process with SpaCy
        doc = self.nlp(cleaned_text)
        
        # Extract entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Get POS tags
        pos_tags = [(token.text, token.pos_) for token in doc]
        
        return {
            'processed_text': cleaned_text,
            'entities': entities,
            'pos_tags': pos_tags
        }

    def _clean_text(self, text: str) -> str:
        """Clean the extracted text."""
        # Remove special characters and extra whitespace
        cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove common OCR artifacts
        cleaned = re.sub(r'[|]', 'I', cleaned)
        cleaned = re.sub(r'[¢]', 'c', cleaned)
        
        return cleaned.strip()

    def summary_generator(self, text:str)->str:
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=200,
                num_return_sequences=1,
                temperature=0.7
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def _get_confidence_scores(self, ocr_data: Dict[str, Any]) -> List[float]:
        """Extract confidence scores for OCR results."""
        return [conf for conf in ocr_data['conf'] if conf != -1]



class EasyOCRProcessor:
    def __init__(self, output_dir: str = "easyocr_extracted_content", languages: List[str] = ['en']):
        """
        Initialize the document processor using EasyOCR.
        
        Args:
            output_dir (str): Directory to save extracted content
            languages (List[str]): List of languages to detect
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.reader = easyocr.Reader(languages)
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        # Load SpaCy model for NLP tasks
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        # Create output directories
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        
    def process_document(self, image_path: str) -> Dict[str, Any]:
        """
        Process a document image to extract text and images.
        
        Args:
            image_path (str): Path to the document image
            
        Returns:
            Dict[str, Any]: Extracted content including text, entities, and image paths
        """
        try:    
            start_time = time.time()        
            # Read and process image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
            
            # Extract text content
            text_result = self.extract_text(image)
            
            # Extract images
            image_result = self.extract_images(image, Path(image_path).stem)
            processing_time = time.time() - start_time
            
            return {
                **text_result,
                'extracted_images': image_result,
                'processing_time': processing_time
            }
            
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    def extract_text(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract and process text using EasyOCR.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Dict[str, Any]: Processed text data
        """
        # Preprocess image
        processed_image = self.preprocess_for_ocr(image)
        
        # Perform OCR with EasyOCR
        results = self.reader.readtext(processed_image)
        
        # Extract text and confidence scores
        text_parts = []
        confidence_scores = []
        bounding_boxes = []
        
        for detection in results:
            bbox, text, score = detection
            text_parts.append(text)
            confidence_scores.append(score)
            bounding_boxes.append(bbox)
        
        # Combine text parts
        raw_text = ' '.join(text_parts)
        
        # Process extracted text
        processed_data = self.process_text(raw_text)
        summary=self.summary_generator(processed_data['processed_text'])
        
        return {
            'raw_text': raw_text,
            'processed_text': processed_data['processed_text'],
            'summary':summary,
            'entities': processed_data['entities'],
            'pos_tags': processed_data['pos_tags'],
            'confidence_scores': confidence_scores,
            'bounding_boxes': bounding_boxes
        }

    def extract_images(self, image: np.ndarray, base_name: str) -> List[Dict[str, Any]]:
        """
        Extract images from document using contour detection.
        
        Args:
            image (np.ndarray): Input image
            base_name (str): Base name for saving extracted images
            
        Returns:
            List[Dict[str, Any]]: List of extracted image information
        """
        extracted = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply different thresholding methods to detect various types of images
        binary_methods = [
            ('otsu', cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),
            ('adaptive', cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2))
        ]
        
        for method_name, binary in binary_methods:
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for idx, contour in enumerate(contours):
                # Filter small contours and check aspect ratio
                if not self._is_valid_image_region(contour, image.shape):
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract region
                roi = image[y:y+h, x:x+w]
                
                # Check if region contains enough variation to be an image
                if not self._is_image_content(roi):
                    continue
                
                # Save extracted region
                image_filename = f"{base_name}_{method_name}_region_{idx + 1}.png"
                image_path = os.path.join(self.images_dir, image_filename)
                
                cv2.imwrite(image_path, roi)
                
                extracted.append({
                    'filename': image_filename,
                    'path': image_path,
                    'position': {'x': x, 'y': y, 'width': w, 'height': h},
                    'area': cv2.contourArea(contour),
                    'method': method_name
                })
        
        return extracted

    def _is_valid_image_region(self, contour: np.ndarray, image_shape: Tuple) -> bool:
        """
        Check if contour is likely to be an image region.
        """
        # Get area and dimensions
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h != 0 else 0
        
        # Calculate relative size
        image_area = image_shape[0] * image_shape[1]
        relative_size = area / image_area
        
        # Define thresholds
        MIN_RELATIVE_SIZE = 0.01  # 1% of image
        MAX_RELATIVE_SIZE = 0.9   # 90% of image
        MIN_ASPECT_RATIO = 0.2
        MAX_ASPECT_RATIO = 5.0
        
        return (MIN_RELATIVE_SIZE <= relative_size <= MAX_RELATIVE_SIZE and 
                MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO)

    def _is_image_content(self, region: np.ndarray, std_threshold: float = 20) -> bool:
        """
        Check if region contains enough variation to be an image.
        """
        gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        return np.std(gray_region) > std_threshold

    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)
        
        # Deskew if needed
        angle = self._get_skew_angle(denoised)
        if abs(angle) > 0.5:
            denoised = self._rotate_image(denoised, angle)
        
        return denoised

    def _get_skew_angle(self, image: np.ndarray) -> float:
        """Calculate skew angle of the image."""
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        return -angle

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate the image by given angle."""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process extracted text using NLP techniques.
        """
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Process with SpaCy
        doc = self.nlp(cleaned_text)
        
        # Extract entities with confidence scores
        entities = [(ent.text, ent.label_, self.nlp(ent.text).vector) 
                   for ent in doc.ents]
        
        # Get POS tags with additional info
        pos_tags = [(token.text, token.pos_, token.tag_, token.dep_) 
                   for token in doc]
        
        return {
            'processed_text': cleaned_text,
            'entities': entities,
            'pos_tags': pos_tags
        }

    def _clean_text(self, text: str) -> str:
        """Clean the extracted text."""
        # Basic cleaning
        cleaned = text.strip()
        
        # Remove special characters while preserving structure
        cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Fix common OCR mistakes
        common_fixes = {
            '|': 'I',
            '¢': 'c',
            '©': 'c',
            '®': 'r',
            '°': 'o',
        }
        for wrong, right in common_fixes.items():
            cleaned = cleaned.replace(wrong, right)
        
        # Fix common patterns
        cleaned = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', cleaned)  # Add space between camelCase
        cleaned = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', cleaned)  # Add space between letters and numbers
        
        return cleaned.strip()
    
    def summary_generator(self,text:str)->str:
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=200,
                num_return_sequences=1,
                temperature=0.7
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

# if __name__ == "__main__":
#     Tprocessor = TesseractProcessor()
#     Eprocessor = EasyOCRProcessor()
    
#     try:
#         # Process a document
#         Tresult = Tprocessor.process_document("dataset/Email/81841302.jpg")
#         Eresult = Eprocessor.process_document("dataset/News/94682942.jpg")
#         # Print performance metrics
#         print("\nPerformance Metrics:")
#         print(f"Processing Time: {Tresult['processing_time']:.2f} seconds")
#         print(f"Processing Time: {Eresult['processing_time']:.2f} seconds")
#         print(f"Confidence Score: {np.mean(Tresult['confidence_scores']):.2f},{np.mean(Eresult['confidence_scores']):.2f}")
        
#         # Print text analysis
#         print("\nExtracted Text Sample:")
#         print(Tresult['processed_text'][:200] + "...")
#         print(Eresult['processed_text'][:200] + "...")
        
#         print("\nNamed Entities:")
#         for entity, label in Tresult['entities']:
#             print(f"{entity}: {label}")
#         print("--------------")
#         for entity, label, _ in Eresult['entities'][:5]:
#             print(f"{entity}: {label}")
        
#         print("\nPOS Tags Sample:")
#         for token, pos in Tresult['pos_tags'][:10]:
#             print(f"{token}: {pos}")
#         print("--------------------")
#         for token, pos, tag, dep in Eresult['pos_tags'][:5]:
#             print(f"{token}: {pos} ({tag}) - {dep}")

#         print("\nExtracted Images:")
#         for img in Tresult['extracted_images']:
#             print(f"Size: {img['size']}")
#             print(f"Saved to: {img['filename']}")
#         print("----------------")
#         for img in Eresult['extracted_images']:
#             print(f"Saved: {img['filename']}")
#             print(f"Method: {img['method']}")
#             print(f"Position: {img['position']}")
#             print("---")
        
#     except Exception as e:
#         print(f"Error: {str(e)}")
