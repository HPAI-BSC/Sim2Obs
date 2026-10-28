# Instructions

1. Download the dataset from:
`/path/to
2. The script to be executed is `main.py`. Update variable `basePath` with the path of the dataset in your local machine.
3. Define your .env file with `ACCESS_KEY` and `SECRET_KEY` variables required to push data to minio. Those variables can be
retrieved from here: https://pass.hpai.bsc.es:18200 using the bucket `space` and `space-read-write` policy.
4. Create and activate a virtual environment with:
- `virtualenv venv` 
- `source venv/bin/activate`
5. Install dependencies:
- `pip install -r requirements.txt`
6. Run `main.py` script.
7. Visit https://storage.hpai.bsc.es:8080/minio/space to see the generated files.