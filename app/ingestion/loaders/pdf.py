import logfire
from pypdf import PdfReader


def parse_pdf(file_path: str) -> str:
    """
    Extract the text from a PDF locally using pypdf.
    Falls back to pdfplumber for pages that yield no text(image heavy pages)
    """
    with logfire.span("PDF Parsing (local)", filename=file_path):
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            logfire.info(f"PDF has {total_pages} pages.")

            text_parts: list[str] = []
            blank_pages: list[int] = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    text_parts.append(text)
                else:
                    blank_pages.append(i + 1)
                    logfire.warning(
                        f"Page {i + 1} yielded no text. It may be image heavy."
                    )

            # Fallback to pdfplumber for blank pages
            if blank_pages:
                logfire.info(
                    f"pypdf yielded no text for pages: {blank_pages}. Falling back to pdfplumber for these pages."
                )
                try:
                    import pdfplumber

                    with pdfplumber.open(file_path) as pdf:
                        for page_num in blank_pages:
                            page = pdf.pages[page_num - 1]
                            text = page.extract_text() or ""
                            if text.strip():
                                text_parts.append(text)
                                logfire.info(
                                    f"Successfully extracted text from page {page_num} using pdfplumber."
                                )
                            else:
                                logfire.warning(
                                    f"Page {page_num} yielded no text even with pdfplumber."
                                )
                except Exception as plumber_err:
                    logfire.warning(
                        f"pdfplumber fallback failed: {plumber_err}. Continuing with available text."
                    )

            full_text = "\n".join(text_parts)
            if not full_text.strip():
                logfire.warning(
                    f"PDF Parsing yielded no text for file: {file_path}. It may be image heavy or encrypted."
                )
            else:
                logfire.info(
                    f"Extracted {len(full_text)} characters of text from {file_path}."
                )

            return full_text
        except Exception as e:
            logfire.error(f"PDF Parse Failed for {file_path}: {e}")
            raise
