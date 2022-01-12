import kfp
from kfp import dsl
from kfp.components import func_to_container_op



@dsl.pipeline(name='advanced-crop-classification-pipeline', description='Classify crops extracted from RPG.')
def crop_classification_pipeline(json_img,shp):

    # create components from yaml manifest 
    download_img = kfp.components.load_component_from_file('process_img/process_img.yaml')
    temporal_stats = kfp.components.load_component_from_file('temporal_stats/temporal_stats.yaml')
    preprocess = kfp.components.load_component_from_file('preprocess_data/preprocess_data.yaml')
    xgboost_classif = kfp.components.load_component_from_file('extreme_gradient_boost/extreme_gradient_boost.yaml')

    # Run first task
    download_task = download_img(json_img,shp)
    #download_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    # create temporal stats from results of the previous task
    temporal_task = temporal_stats(download_task.output)
    temporal_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    # preprocess data 
    preprocess_task = preprocess(temporal_task.output)
    preprocess_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    # classification with XGBoost
    xgboost_task = xgboost_classif(data=preprocess_task.output)
    xgboost_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
   



if __name__ == '__main__':
    kfp.compiler.Compiler().compile(crop_classification_pipeline, 'advanced-crop-classification-pipeline.yaml')
