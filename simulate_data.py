from random import  uniform
from time import sleep
import mlrun

def invoke(sample_cnt=1000, project_name="ny-lgbm-demo-yaronh"):

    # Load serving function
    project = mlrun.get_or_create_project(project_name)
    serving_function = project.get_function("serving")

    # Load the dataset
    data = mlrun.get_dataitem("https://s3.us-east-1.wasabisys.com/iguazio/data/nyc-taxi/test.csv").as_df()

    # Sending random requests
    for i in range(sample_cnt):
        data_point = data.iloc[i].to_dict()
        try:
            resp = serving_function.invoke(path='/predict', body=data_point)
            print(resp)
            sleep(uniform(0.02, 0.03))
        except OSError:
            pass

if __name__ == '__main__':
    invoke()