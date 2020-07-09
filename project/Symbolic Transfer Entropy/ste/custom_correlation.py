import pandas as pd

# Python module.
#
#
# Pietro Foini
#
# Year: 2020

def corr_pairwise(df, method = None):
    # Define the correlation matrix to fill with values.
    CORR = pd.DataFrame(index = df.columns, columns = df.columns)
    
    def interation_1(serie):
        source = serie
        
        def interation_2(serie, source):
            target = serie
            correlation = method(source, target)
            CORR[source.name].loc[target.name] = correlation

        df.apply(interation_2, args = [source])
     
    df.apply(interation_1)
    
    return CORR