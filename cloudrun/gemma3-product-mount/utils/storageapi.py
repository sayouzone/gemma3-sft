# -*- coding: utf-8 -*-
#from google.cloud import storage, transfer_manager
from google.cloud.storage import Client, transfer_manager

import os
import time
from dotenv import load_dotenv

#https://cloud.google.com/storage/docs/samples/storage-transfer-manager-download-bucket?hl=ko#code-sample

class StorageAPI:

    def __init__(self, bucket_name):
        # 스토리지 클라이언트 초기화
        self.bucket_name = bucket_name
        self.storage_client = Client()
        self.bucket = self.storage_client.get_bucket(bucket_name)
        #self.bucket = self.storage_client.bucket(bucket_name)

    def download_gcs_folder(self, source_folder, destination_folder):
        """GCS 버킷의 폴더를 로컬 디렉터리로 다운로드합니다.

        Args:
            bucket_name (str): GCS 버킷 이름 (예: 'my-bucket')
            source_folder (str): 다운로드할 GCS 내 폴더 경로 (예: 'data/images/')
            destination_folder (str): 파일을 저장할 로컬 디렉터리 경로 (예: './downloaded_images')
        """
        
        start_time = time.time()
        # 지정된 폴더(prefix)에 있는 모든 blob(파일) 목록 가져오기ß
        blobs = self.bucket.list_blobs(prefix=source_folder)

        # 로컬에 저장할 디렉터리가 없으면 생성
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            print(f"'{destination_folder}' 디렉터리를 생성했습니다.")

        print(f"'{source_folder}' 폴더를 '{destination_folder}'(으)로 다운로드 시작.")
        for blob in blobs:
            # 다운로드할 파일의 전체 로컬 경로 생성
            # blob.name은 'data/images/cat.jpg'와 같은 전체 경로입니다.
            destination_path = os.path.join(destination_folder, os.path.basename(blob.name))

            # 파일 다운로드
            try:
                # 빈 폴더 자체를 나타내는 blob은 건너뜁니다.
                if blob.name.endswith('/'):
                    continue
            
                blob.download_to_filename(destination_path)
                print(f"'{destination_path}'(으)로 다운로드")
            except Exception as e:
                print(f"'{blob.name}' 파일 다운로드 중 오류 발생: {e}")
        
        end_time = time.time()
        print(f'Elapsed time: {(end_time - start_time)}s')

    def download_folder(self, folder, dest_dir):
        workers=8
        max_results=1000
        blob_names = [blob.name for blob in self.bucket.list_blobs(
            prefix=folder, max_results=max_results
        )]
        print(blob_names)
        start_time = time.time()
        results = transfer_manager.download_many_to_path(
            self.bucket, blob_names, destination_directory=dest_dir, max_workers=workers
        )

        for name, result in zip(blob_names, results):
            # The results list is either `None` or an exception for each blob in
            # the input list, in order.

            if isinstance(result, Exception):
                print("Failed to download {} due to exception: {}".format(name, result))
            #else:
            #    print("Downloaded {} to {}.".format(name, dest_dir + name))
        end_time = time.time()
        print(f'Elapsed time: {(end_time - start_time)}s')

    def download_chunks_concurrently(
        self, blob_name, filename, chunk_size=32 * 1024 * 1024, workers=8
    ):
        """
        Download a single file in chunks, concurrently in a process pool.
        https://cloud.google.com/storage/docs/samples/storage-transfer-manager-download-chunks-concurrently?hl=ko
        """

        # The ID of your GCS bucket
        # bucket_name = "your-bucket-name"

        # The file to be downloaded
        # blob_name = "target-file"

        # The destination filename or path
        # filename = ""

        # The size of each chunk. The performance impact of this value depends on
        # the use case. The remote service has a minimum of 5 MiB and a maximum of
        # 5 GiB.
        # chunk_size = 32 * 1024 * 1024 (32 MiB)

        # The maximum number of processes to use for the operation. The performance
        # impact of this value depends on the use case, but smaller files usually
        # benefit from a higher number of processes. Each additional process occupies
        # some CPU and memory resources until finished. Threads can be used instead
        # of processes by passing `worker_type=transfer_manager.THREAD`.
        # workers=8
        
        blob = self.bucket.blob(blob_name)
        transfer_manager.download_chunks_concurrently(
            blob, filename, chunk_size=chunk_size, max_workers=workers
        )

        print("Downloaded {} to {}.".format(blob_name, filename))

if __name__ == "__main__":
    # .env 파일 로드
    load_dotenv()

    ft_local_path = os.getenv("ft_local_path")
    ft_gcs_bucket = os.getenv("ft_gcs_bucket")
    ft_gcs_path = os.getenv("ft_gcs_path")
    print("ft_local_path", ft_local_path)
    print("ft_gcs_bucket", ft_gcs_bucket)
    print("ft_gcs_path", ft_gcs_path)

    start_time = time.time()
    storage_api = StorageAPI(ft_gcs_bucket)
    #storage_api.download_folder(ft_gcs_path, ".")
    storage_api.download_gcs_folder(ft_gcs_path, "./model")
    end_time = time.time()
    print(f'Total elapsed time: {(end_time - start_time)}s')