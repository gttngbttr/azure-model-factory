from azure.storage.blob import ContainerClient
from datetime import datetime, date, timezone
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--scoring_datastore", type=str, default=None)
    parser.add_argument("--score_container", type=str, default=None)
    parser.add_argument("--scoring_datastore_key", type=str, default=None)
    parser.add_argument("--scoring_output_filename", type=str, default=None)

    return parser.parse_args()


def copy_output(args):
    print("Output : {}".format(args.output_path))

    accounturl = "https://{}.blob.core.windows.net".format(
        args.scoring_datastore
    )  # NOQA E501

    containerclient = ContainerClient(
        accounturl, args.score_container, args.scoring_datastore_key
    )

    destfolder = date.today().isoformat()
    filetime = (
        datetime.now(timezone.utc)
        .time()
        .isoformat("milliseconds")
        .replace(":", "_")
        .replace(".", "_")
    )  # noqa E501
    destfilenameparts = args.scoring_output_filename.split(".")
    destblobname = "{}/{}_{}.{}".format(
        destfolder, destfilenameparts[0], filetime, destfilenameparts[1]
    )

    destblobclient = containerclient.get_blob_client(destblobname)
    with open(
        os.path.join(args.output_path, "parallel_run_step.txt"), "rb"
    ) as scorefile:  # noqa E501
        destblobclient.upload_blob(scorefile, blob_type="BlockBlob")


if __name__ == "__main__":
    args = parse_args()
    if (
        args.scoring_datastore is None
        or args.scoring_datastore.strip() == ""
        or args.score_container is None
        or args.score_container.strip() == ""
        or args.scoring_datastore_key is None
        or args.scoring_datastore_key.strip() == ""
        or args.scoring_output_filename is None
        or args.scoring_output_filename.strip() == ""
        or args.output_path is None
        or args.output_path.strip() == ""
    ):
        print("Missing parameters")
    else:
        copy_output(args)
