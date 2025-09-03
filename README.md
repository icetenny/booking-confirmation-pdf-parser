# booking-confirmation-pdf-parser


## SetUp
### 1. Clone this repository
```
https://github.com/icetenny/booking-confirmation-pdf-parser.git
```
### 2. Navigate into the working folder

```
cd booking-confirmation-pdf-parser
```

### 3. Create a new Python environment 
#### With Python VENV
```
python -m venv env_pdf
source env_pdf/bin/activate   # (Linux / macOS)
env_pdf\Scripts\activate      # (Windows PowerShell / CMD)
```

#### With Conda
```
conda create -n env_pdf python=3.11
conda activate env_pdf
```

### 4. Install required libraries and the dependencies

```
pip install -r requirements.txt
```

## Run `main.py` with `PATH_TO_PDF_FILE` as argument
```
python main.py {PATH_TO_PDF_FILE}

# Example
python main.py pdfs/BookingConfirm-SE.pdf
```

