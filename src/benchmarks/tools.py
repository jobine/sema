import os
import json
import requests
from ..utils import get_logger
from tqdm import tqdm


logger = get_logger(__name__)


def download_file(url: str, destination_path: str, max_retries: int = 3, timeout: int = 10) -> None:
    '''
    Download a file from a URL to a local destination.
    '''
    
    # if destination directory does not exist, create it
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    for attempt in range(max_retries):
        try:
            logger.info(f'Downloading {url} to {destination_path} (Attempt {attempt + 1}/{max_retries})')
            
            resume_pos = 0
            if os.path.exists(destination_path):
                resume_pos = os.path.getsize(destination_path)
                logger.info(f'Resuming download from byte position {resume_pos}')

            resp_head = requests.head(url, timeout=timeout)
            total_size = int(resp_head.headers.get('Content-Length', 0))

            if resume_pos < total_size:
                headers = {'Range': f'bytes={resume_pos}-'} if resume_pos > 0 else {}
                with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
                    r.raise_for_status()
                    mode = 'ab' if resume_pos > 0 else 'wb'
                    
                    with tqdm(total=total_size, initial=resume_pos, unit='iB', unit_scale=True, desc=os.path.basename(destination_path)) as pbar, open(destination_path, mode) as f:
                        for chunk in r.iter_content(chunk_size=1024):
                            if chunk:
                                size = f.write(chunk)
                                pbar.update(size)
            
            logger.info(f'Successfully downloaded {url} to {destination_path}')
            return
        except Exception as e:
            logger.error(f'Error downloading {url}: {e}')
            if attempt == max_retries - 1:
                logger.error(f'Failed to download {url} after {max_retries} attempts.')
                raise
            else:
                logger.info(f'Retrying downloading {url}...')

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    with open(destination_path, 'wb') as f:
        f.write(response.content)


def load_json(file_path: str) -> list[dict] | dict:
    '''
    Load a JSON or JSONL file and return its content as a list of dictionaries.
    '''

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File not found: {file_path}')
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.jsonl'):
            for line in f:
                data.append(json.loads(line))
        elif file_path.endswith('.json'):
            data = json.load(f)
        else:
            raise ValueError(f'Unsupported file format: {file_path}')
    return data


# Example usage
if __name__ == '__main__':
    # download_file(url='http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json', destination='data/hotpot_dev_distractor_v1.json')
    download_file(url='https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl', destination_path='data/gsm8k_train.jsonl')