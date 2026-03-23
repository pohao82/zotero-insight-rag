import logging
from langchain_community.document_loaders import PyPDFLoader

class DocumentParser:
    def __init__(self, mode="marker", use_gpu=True):
        """
        mode: 'marker' (Layout-aware, markdown) or 'pypdf' (Plain text)
        use_gpu: Only affects 'marker' mode. (need cuda)
        """
        self.mode = mode
        #self.use_gpu = use_gpu
        self.converter = None

        if self.mode == "marker":
            try:
                from marker.converters.pdf import PdfConverter
                from marker.models import create_model_dict

                device = 'cuda:0' if use_gpu else 'cpu'
                self.converter = PdfConverter(
                    artifact_dict=create_model_dict(device=device)
                )
                print(f"initialized Marker-PDF on {device}")
            except ImportError:
                logging.warning("Marker-PDF not found. Falling back to PyPDF mode.")
                self.mode = "pypdf"

    # public interface
    def parse_to_text(self, pdf_path):
        """Standard entry point for benchmarking both modes."""
        if self.mode == "marker" and self.converter:
            return self._parse_with_marker(pdf_path)
        return self._parse_with_pypdf(pdf_path)

    def _parse_with_marker(self, pdf_path):
        from marker.output import text_from_rendered
        # Marker handles complex layouts, equations, and tables
        rendered = self.converter(str(pdf_path))
        full_text, _, _ = text_from_rendered(rendered)
        return full_text

    def _parse_with_pypdf(self, pdf_path):
        # Baseline: Fast, but loses table structure and reading order
        loader = PyPDFLoader(str(pdf_path))
        # PyPDF return pages, need to join
        pages = loader.load()
        return "\n\n".join([p.page_content for p in pages])
