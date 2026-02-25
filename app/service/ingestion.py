import os
from pypdf import PdfReader
from core.logging import logger
from typing import List

class IngestionService:
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extracts all text from a PDF file.
        """
        if not os.path.exists(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            raise FileNotFoundError(f"File not found: {pdf_path}")
        
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            logger.info(f"Successfully extracted text from {pdf_path}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise e

    @staticmethod
    def extract_text_from_bytes(pdf_bytes: bytes) -> str:
        """
        Extracts all text from PDF bytes.
        """
        from io import BytesIO
        try:
            reader = PdfReader(BytesIO(pdf_bytes))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from bytes: {str(e)}")
            raise e
