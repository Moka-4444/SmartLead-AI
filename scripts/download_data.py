import os
import urllib.request
import zipfile
import pathlib

def download_and_extract():
    project_root = pathlib.Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "bank-additional-full.csv"
    
    if csv_path.exists():
        print(f"Dataset already exists at {csv_path}")
        return
        
    url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
    zip_path = data_dir / "bank_marketing.zip"
    
    print(f"Downloading dataset from {url}...")
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting archive...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(data_dir)
        
    # The zip contains bank.zip and bank-additional.zip inside!
    additional_zip = data_dir / "bank-additional.zip"
    if additional_zip.exists():
        print("Extracting bank-additional.zip...")
        with zipfile.ZipFile(additional_zip, 'r') as zf:
            zf.extractall(data_dir)
            
    # Check if bank-additional-full.csv is in bank-additional/
    extracted_csv = data_dir / "bank-additional" / "bank-additional-full.csv"
    if extracted_csv.exists():
        # Copy to data root
        import shutil
        shutil.copy(extracted_csv, csv_path)
        print(f"Successfully placed bank-additional-full.csv at {csv_path}")
    else:
        # Search for it
        found = list(data_dir.glob("**/bank-additional-full.csv"))
        if found:
            import shutil
            shutil.copy(found[0], csv_path)
            print(f"Found and copied from {found[0]} to {csv_path}")
        else:
            raise FileNotFoundError("Could not locate bank-additional-full.csv after extraction.")
            
    # Clean up zips if desired
    if zip_path.exists():
        zip_path.unlink()
    if additional_zip.exists():
        additional_zip.unlink()

if __name__ == "__main__":
    download_and_extract()
