
import pandas as panda


def FileProcessing(path,FileName):
    # load the excel sheet
    my_data_df=None
    my_data_df = panda.read_csv(path+FileName)
    # The data has been loaded in to the dataFrame
    #print my_data_df.head()
    my_data_df = my_data_df.drop('bruises?-no', 1)
    #print my_data.head()
    my_data_df['bruises?-bruises-Copy']= my_data_df['bruises?-bruises']
    my_data_df = my_data_df.drop('bruises?-bruises', 1)
    my_data_df['bruises?-bruises']= my_data_df['bruises?-bruises-Copy']
    my_data_df = my_data_df.drop('bruises?-bruises-Copy', 1)
    #print my_data_df.head()
    my_data = my_data_df.values.tolist()
    #print my_data[1:2]
    return my_data_df