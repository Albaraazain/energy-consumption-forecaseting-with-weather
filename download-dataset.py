import kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files('jeanmidev/smart-meters-in-london', path='.', unzip=True)
