import logfire
from unstructured.partition.auto import partition


def parse_office(file_path: str):
    """
    Parses office documents(.docx,.pptx) using Unstructured library.
    """
    with logfire.span("Office document parsing", filename=file_path):
        try:
            elements = partition(filename=file_path)
            full_text = "\n".join([str(el) for el in elements])

            if not full_text.strip():
                logfire.warning(f"Unstructured returned empty text for {file_path}")
            else:
                logfire.info(f"Successfully parsed {len(full_text)} characters")
            return full_text
        except Exception as e:
            logfire.error(f"Office parse failed:{e}")
            raise e
