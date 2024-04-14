from langchain.pydantic_v1 import BaseModel, Field
from langchain.document_loaders import (
    WebBaseLoader,
    SeleniumURLLoader,
    UnstructuredURLLoader,
)
from langchain.tools import BaseTool, tool
from typing import Optional, Type
from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
)
import requests
import os
import requests
from tqdm import tqdm
import PyPDF2

class SearchInput(BaseModel):
    url: str = Field(description="should be a url")


class CustomWebScrapTool(BaseTool):
    name = "custom_web_loader"
    description = "Load/scrap all the data from the provided url"
    args_schema: Type[BaseModel] = SearchInput

    def _run(
        self, url: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        self.download_drive_link(url,"./a.pdf")
        data=self.read_pdf("./a.pdf")
        # print(data.content)
        return data

    def download_drive_link(self,url,path):
        # Extract the file ID from the Google Drive link
        file_id = url.split("/")[-2]

        # URL for downloading the file
        download_url = f"https://docs.google.com/uc?export=download&id={file_id}"

        # Send a request to initiate the download
        session = requests.Session()
        response = session.get(download_url, stream=True)

        # Get the total file size
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

        # Create the file path if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        try:
            # Download the file and save it to the specified path
            with open(path, "wb") as file:
                for data in response.iter_content(block_size):
                    if not data:
                        break
                    file.write(data)
                    progress_bar.update(len(data))
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            progress_bar.close()
            session.close()

        if total_size != 0 and progress_bar.n != total_size:
            print("Error: Download incomplete")
        else:
            print("File downloaded successfully")

    def read_pdf(self,path)-> str:
        print("on 75")
        # Open the PDF file
        with open(path, "rb") as file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            # Get the number of pages in the PDF
            num_pages = len(pdf_reader.pages)
            # Iterate over each page and extract the text
            text = ""
            for page_num in range(num_pages):
                # Get the page object
                page = pdf_reader.pages[page_num]
                # Extract the text from the page
                page_text = page.extract_text()
                # Append the page text to the overall text
                text += page_text
        return text


# Instantiate the ReadFileTool
url_scraper_tool = CustomWebScrapTool()


def generate_function_config(tool):
    # Define the function schema based on the tool's args_schema
    function_schema = {
        "name": tool.name.lower().replace(" ", "_"),
        "description": tool.description,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": ["url"],
        },
    }

    if tool.args is not None:
        function_schema["parameters"]["properties"] = tool.args

    return function_schema
