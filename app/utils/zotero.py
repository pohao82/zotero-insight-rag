import requests
from pathlib import Path

class ZoteroLocalClient:
    def __init__(self, storage_path="/workspace/Zotero/storage", port=23119):
        self.storage_path = Path(storage_path)
        self.api_url = f"http://localhost:{port}/api/users/0/items"

    def fetch_all_items(self):
        """
        Fetches all items from the local Zotero API.
        Not per items document. All items also other objects including annotations. 

        """
        response = requests.get(self.api_url, params={"format": "json"})
        response.raise_for_status()
        return response.json()

    # for local files
    def map_parent_to_pdf(self, all_items):
        """
        Creates a map of Parent Key -> Physical PDF Path.
        Each file has its own subdirectory named with an 8-character string.
        """
        parent_to_pdf = {}
        for item in all_items:
            # data stores also annotation with another key something like 'NDN3H4ZP'
            data = item.get('data', {})
            # Look for attachment items that are PDFs
            if data.get('itemType') == 'attachment' and data.get('contentType') == 'application/pdf':
                parent_key = data.get('parentItem')
                attachment_key = item.get('key') # The folder name

                item_folder = self.storage_path / attachment_key
                pdf_files = list(item_folder.glob("*.pdf"))

                if pdf_files and parent_key:
                    parent_to_pdf[parent_key] = pdf_files[0]
        return parent_to_pdf

    def get_library_metadata(self):
        all_items = self.fetch_all_items()
        parent_to_pdf = self.map_parent_to_pdf(all_items)

        library_data = []

        for item in all_items:
            data = item.get('data', {})
            item_type = data.get('itemType')

            # Skip attachments/notes to focus on the actual papers/books
            if item_type in ['attachment', 'note']:
                continue

            item_key = item.get('key')

            # Extracting specific metadata fields
            metadata = {
                "key": item_key,
                "title": data.get('title'),
                "date": data.get('date'),
                "journal": data.get('publicationTitle') or data.get('proceedingsTitle'),
                # Creators is a list of dicts: [{'firstName': 'John', 'lastName': 'Doe', 'creatorType': 'author'}]
                "authors": [f"{c.get('firstName')} {c.get('lastName')}" for c in data.get('creators', [])],
                "pdf_path": str(parent_to_pdf.get(item_key, "No PDF found"))
            }
    
            library_data.append(metadata)
    
        return library_data
