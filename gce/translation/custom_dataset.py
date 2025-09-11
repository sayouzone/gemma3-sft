import datasets

class CustomDataset(datasets.GeneratorBasedBuilder):
    """나만의 커스텀 데이터셋"""

    def _info(self):
        return datasets.DatasetInfo(
            description="이 데이터셋은 텍스트와 레이블로 구성됩니다.",
            features=datasets.Features({
                "text": datasets.Value("string"),
                "label": datasets.ClassLabel(names=["부정", "긍정"]),
            }),
            homepage="https://my-dataset-homepage.com",
            license_name="Apache-2.0",
        )

    def _split_generators(self, dl_manager):
        # 데이터 다운로드 (필요 시)
        # data_dir = dl_manager.download_and_extract("URL_TO_YOUR_DATA")
        data_dir = "." # 로컬 파일 사용 예시

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": f"{data_dir}/train.jsonl"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": f"{data_dir}/test.jsonl"},
            ),
        ]

    def _generate_examples(self, filepath):
        """데이터 파일에서 실제 샘플을 생성합니다."""
        with open(filepath, "r", encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                yield id_, {
                    "text": data["text"],
                    "label": data["label"],
                }