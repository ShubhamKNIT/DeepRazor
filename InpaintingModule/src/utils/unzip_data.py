from zipfile import ZipFile
def unzip_data(file_path, extract_path):
    with ZipFile(file_path, 'r') as zip:
        zip.extractall(extract_path)
        zip.close()