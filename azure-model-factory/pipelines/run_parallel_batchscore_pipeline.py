from azure.storage.blob import ContainerClient
from aml_services.utils.env_variables import Env
from azureml.core import Experiment, Workspace
from azureml.pipeline.core import PublishedPipeline
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_id", type=str, default=None)
    return parser.parse_args()


def get_pipeline(pipeline_id, ws: Workspace, env: Env):
    if pipeline_id is not None:
        scoringpipeline = PublishedPipeline.get(ws, pipeline_id)
    else:
        pipelines = PublishedPipeline.list(ws)
        scoringpipelinelist = [
            pl for pl in pipelines if pl.name == env.scoring_pipeline_name
        ]  # noqa E501

        if scoringpipelinelist.count == 0:
            raise Exception(
                "No pipeline found matching name:{}".format(env.scoring_pipeline_name)  # NOQA: E501
            )
        else:
            # latest published
            scoringpipeline = scoringpipelinelist[0]

    return scoringpipeline


def copy_output(step_id: str, env: Env):
    accounturl = "https://{}.blob.core.windows.net".format(
        env.scoring_datastore_storage_name
    )

    srcblobname = "azureml/{}/{}_out/parallel_run_step.txt".format(
        step_id, env.scoring_datastore_storage_name
    )

    srcbloburl = "{}/{}/{}".format(
        accounturl, env.scoring_datastore_output_container, srcblobname
    )

    containerclient = ContainerClient(
        accounturl,
        env.scoring_datastore_output_container,
        env.scoring_datastore_access_key,
    )
    srcblobproperties = containerclient.get_blob_client(
        srcblobname
    ).get_blob_properties()  # noqa E501

    destfolder = srcblobproperties.last_modified.date().isoformat()
    filetime = (
        srcblobproperties.last_modified.time()
        .isoformat("milliseconds")
        .replace(":", "_")
        .replace(".", "_")
    )  # noqa E501
    destfilenameparts = env.scoring_datastore_output_filename.split(".")
    destblobname = "{}/{}_{}.{}".format(
        destfolder, destfilenameparts[0], filetime, destfilenameparts[1]
    )

    destblobclient = containerclient.get_blob_client(destblobname)
    destblobclient.start_copy_from_url(srcbloburl)


def run_batchscore_pipeline():
    try:
        env = Env()

        args = parse_args()

        aml_workspace = Workspace.get(
            name=env.workspace_name,
            subscription_id=env.subscription_id,
            resource_group=env.resource_group,
        )

        scoringpipeline = get_pipeline(args.pipeline_id, aml_workspace, env)

        experiment = Experiment(workspace=aml_workspace, name=env.experiment_name)  # NOQA: E501

        run = experiment.submit(
            scoringpipeline,
            pipeline_parameters={
                "model_name": env.model_name,
                "model_version": env.model_version,
                "model_tag_name": " ",
                "model_tag_value": " ",
            },
        )

        run.wait_for_completion(show_output=True)

        if run.get_status() == "Finished":
            copy_output(list(run.get_steps())[0].id, env)

    except Exception as ex:
        print("Error: {}".format(ex))


if __name__ == "__main__":
    run_batchscore_pipeline()
