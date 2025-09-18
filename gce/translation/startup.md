
#### VM instances 목록

```bash
gcloud compute instances list --project sayouzone-ai
```

```bash
NAME                 ZONE           MACHINE_TYPE   PREEMPTIBLE  INTERNAL_IP  EXTERNAL_IP  STATUS
gemma3-g2s4-l4-test  us-central1-c  g2-standard-4               10.128.0.2                TERMINATED
```

#### VM instance 생성

```bash
PROJECT_ID=sayouzone-ai
INSTANCE_NAME=gemma3-g2s4-l4-test
REGION=us-central1
ZONE=us-central1-c
BUCKET_NAME=sayouzone-ai-gemma3
```

#### VM instance 시작

```bash
gcloud compute instances start $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE
```

#### VM instance 중지

```bash
gcloud compute instances stop $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE
```

#### Connect GCE VM Instance

```bash
gcloud compute ssh $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE
```

#### Copy local file to GCE VM Instance using scp

```bash
LOCAL_FILE_PATH=datasets/The_Wonderful_Wizard_of_Oz_1.csv
DESTINATION_PATH=/gemma3/translation/datasets
REMOTE_ACCOUNT=sjkim

gcloud compute scp $LOCAL_FILE_PATH $REMOTE_ACCOUNT@$INSTANCE_NAME:~$DESTINATION_PATH \
    --project=$PROJECT_ID \
    --zone=$ZONE
```

```bash
LOCAL_FILE_PATH=finetune_translation_genre_12b.py
DESTINATION_PATH=/gemma3/translation
REMOTE_ACCOUNT=sjkim

gcloud compute scp $LOCAL_FILE_PATH $REMOTE_ACCOUNT@$INSTANCE_NAME:~$DESTINATION_PATH \
    --project=$PROJECT_ID \
    --zone=$ZONE
```

Multiple files

```bash
DESTINATION_PATH=/gemma3/text-to-sql
REMOTE_ACCOUNT=sjkim

gcloud compute scp text-to-sql/finetune_sql_sfttrainer.py text-to-sql/load_dataset.py $REMOTE_ACCOUNT@$INSTANCE_NAME:~$DESTINATION_PATH \
    --project=$PROJECT_ID \
    --zone=$ZONE
```

Folder

```bash
LOCAL_FILE_PATH=text-to-sql
DESTINATION_PATH=/gemma3/text-to-sql
REMOTE_ACCOUNT=sjkim

gcloud compute scp --recurse $LOCAL_FOLDER $REMOTE_ACCOUNT@$INSTANCE_NAME:~$DESTINATION_PATH \
    --project=$PROJECT_ID \
    --zone=$ZONE
```

#### Run Locallyc

```bash
export GOOGLE_APPLICATION_CREDENTIALS=~/Development/sayouzone/credentials/sayouzone-ai-ce33a3d8d424.json
```

