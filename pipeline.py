import kfp
from kfp import dsl
from kfp.components import create_component_from_func





@dsl.pipeline(name='advanced-crop-classification-pipeline', description='Classify crops extracted from RPG.')
def crop_classification_pipeline(json_img: str,shp:str,cross_validation: bool = False,iterations: int = 2,cv: int = 2):

    def compare_models(xgboost_csv : str, lstm_csv : str) -> str:
        
        xgb_acc = float(xgboost_csv.split('accuracy,')[1].split(',')[0])

        lstm_acc = float(lstm_csv.split('accuracy,')[1].split(',')[0])
        
        if xgb_acc>=lstm_acc:
            print ("XGBoost model will be used for serving")
            #return xgboost_model
            return "XGB"
        else:
            print ("LSTM will be used for serving")
            #return lstm_model
            return "LSTM"

    # create components from yaml manifest 
    download_img = kfp.components.load_component_from_file('process_img/process_img.yaml')
    temporal_stats = kfp.components.load_component_from_file('temporal_stats/temporal_stats.yaml')
    preprocess = kfp.components.load_component_from_file('preprocess_data/preprocess_data.yaml')
    xgboost_classif = kfp.components.load_component_from_file('extreme_gradient_boost/extreme_gradient_boost.yaml')
    lstm_classif = kfp.components.load_component_from_file('lstm/lstm.yaml')
    compare = create_component_from_func(
                        func=compare_models,
                        base_image='python:3.7', 
                        #output_component_file='compare_models.yaml', 
                        #packages_to_install=['pandas==0.24'],
                    )

    # Run first task
    download_task = download_img(json_img,shp)
    #download_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    # create temporal stats from results of the previous task
    temporal_task = temporal_stats(download_task.output)
    #temporal_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    # preprocess data 
    preprocess_task = preprocess(temporal_task.output)
    #preprocess_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    # classification with XGBoost
    xgboost_task = xgboost_classif(preprocess_task.output,cross_validation,iterations,cv)
    #xgboost_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    # classification with LSTM
    lstm_task = lstm_classif(preprocess_task.output,cross_validation,iterations,cv)
    #lstm_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    # compare models
    compare_task = compare(xgboost_task.outputs['Report'],lstm_task.outputs['Report'])
    compare_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
   



if __name__ == '__main__':
    kfp.compiler.Compiler().compile(crop_classification_pipeline, 'advanced-crop-classification-pipeline.yaml')
