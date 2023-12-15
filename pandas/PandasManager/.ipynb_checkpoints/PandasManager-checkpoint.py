import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PandasManager:
    dataframe = None
    
    def __init__(self):
        self.dataframe = None
    
    ### importar o dataframe por um path de um arquivo
    @classmethod
    def importDataframeFromURL(cls, path, sep = ','):
        df = pd.read_csv(path, sep=sep)   
        cls.dataframe = df
        return df

    ### Retorna uma lista com o nome das colunas do DataFrame
    @classmethod
    def getColumns(cls, df):
        return cls.dataframe.columns.values

    ### Retorna um dicionário que mapeia o nome da coluna para o tipo de dados correspondente
    @classmethod
    def getTypes(cls):
        lista = {}
        for nome_coluna, tipo in cls.dataframe.dtypes.iteritems():
            # print(f'Coluna: {nome_coluna}, Tipo: {tipo}')
            lista[nome_coluna] = str(tipo)
        return lista

    ### Filtra o DataFrame com base nos valores de uma coluna, permitindo correspondência insensível a maiúsculas/minúsculas se necessário
    @classmethod
    def filtrarPor(cls, coluna, valor, tolerancia=1):
        resultado = cls.dataframe[cls.dataframe[coluna].str.contains(valor, case=False, na=False)]
        return resultado
    
    ### Retorna estatísticas descritivas para o DataFrame, como contagem, média, desvio padrão, etc
    @classmethod
    def describe(cls):
        return cls.dataframe.describe()

    ###  Permite remover colunas do DataFrame
    @classmethod
    def dropColumns(cls, columns):
        cls.dataframe = cls.dataframe.drop(columns, axis=1)
        return cls.dataframe
    
    ### Permite renomear as colunas do DataFrame com base em um dicionário de mapeamento
    @classmethod
    def renameColumns(cls, column_mapping):
        cls.dataframe.rename(columns=column_mapping, inplace=True)
        return cls.dataframe

    ### Classifica o DataFrame com base em uma coluna específica, permitindo ordenação ascendente ou descendente
    @classmethod
    def sortByColumn(cls, column, ascending=True):
        cls.dataframe = cls.dataframe.sort_values(by=column, ascending=ascending)
        return cls.dataframe

    ### Agrupa o DataFrame com base em uma coluna específica
    @classmethod
    def groupBy(cls, group_column):
        return cls.dataframe.groupby(group_column)

    ### Cria uma tabela dinâmica com base nas colunas de índice, colunas e valores especificados
    @classmethod
    def pivotTable(cls, index, columns, values):
        return cls.dataframe.pivot_table(index=index, columns=columns, values=values)

    ### Reseta o índice do DataFrame após realizar operações que possam ter modificado a ordem das linhas
    @classmethod
    def resetIndex(cls):
        cls.dataframe = cls.dataframe.reset_index(drop=True)
        return cls.dataframe
 
    ### Permite adicionar uma nova coluna ao DataFrame com os valores especificados
    @classmethod
    def addColumn(cls, column_name, values):
        cls.dataframe[column_name] = values
        return cls.dataframe

    ### Remove linhas duplicadas com base em um subconjunto opcional de colunas
    @classmethod
    def dropDuplicates(cls, subset=None):
        cls.dataframe = cls.dataframe.drop_duplicates(subset=subset)
        return cls.dataframe

    ### Aplica uma função específica a uma coluna do DataFrame
    @classmethod
    def applyFunction(cls, column, func):
        cls.dataframe[column] = cls.dataframe[column].apply(func)
        return cls.dataframe

    ### Preenche os valores ausentes (NaN) no DataFrame com um valor específico
    @classmethod
    def fillMissingValues(cls, value):
        cls.dataframe = cls.dataframe.fillna(value)
        return cls.dataframe
    
    ### Um método que permite selecionar linhas com base em uma condição específica. Por exemplo, você poderia selecionar todas as linhas onde o valor de uma coluna seja maior que um determinado limite
    @classmethod
    def selectRowsByCondition(cls, condition):
        selected_rows = cls.dataframe[condition]
        return selected_rows

    ### Adicione métodos para calcular estatísticas personalizadas, além das estatísticas descritivas padrão. Por exemplo, você poderia calcular a mediana, a moda ou qualquer outra estatística personalizada
    @classmethod
    def calculateCustomStatistics(cls, custom_function):
        custom_stats = custom_function(cls.dataframe)
        return custom_stats

    ### Adicione métodos para criar gráficos a partir do DataFrame usando bibliotecas como Matplotlib ou Seaborn. Isso pode incluir gráficos de barras, gráficos de dispersão, histogramas, entre outros
    @classmethod
    def createPlot(cls, x_column, y_column, plot_type='scatter'):
        import matplotlib.pyplot as plt

        if plot_type == 'scatter':
            plt.scatter(cls.dataframe[x_column], cls.dataframe[y_column])
        elif plot_type == 'bar':
            plt.bar(cls.dataframe[x_column], cls.dataframe[y_column])
        # Adicione mais tipos de gráficos conforme necessário

        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f'{plot_type} Plot')
        plt.show()

    ### Adicione métodos para exportar o DataFrame para diferentes formatos, como Excel, CSV, JSON, etc
    @classmethod
    def exportToCSV(cls, filename):
        cls.dataframe.to_csv(filename, index=False)

    ### Métodos para aplicar transformações de dados ao DataFrame, como escalonamento, normalização ou codificação de variáveis categóricas 
    @classmethod
    def applyDataTransformation(cls, transformation_function):
        transformed_data = transformation_function(cls.dataframe)
        return transformed_data

    ### Se você estiver trabalhando com dados temporais, considere adicionar métodos para realizar resampling, calcular médias móveis ou lidar com séries temporais
    @classmethod
    def handleTimeSeriesData(cls, time_column, frequency):
        time_series_data = cls.dataframe.set_index(pd.to_datetime(cls.dataframe[time_column]))
        resampled_data = time_series_data.resample(frequency).sum()
        return resampled_data

    ### Métodos para realizar amostragem aleatória ou estratificada do DataFrame
    @classmethod
    def sampleData(cls, fraction):
        sampled_data = cls.dataframe.sample(frac=fraction)
        return sampled_data

    ### Métodos para unir dois DataFrames com base em chaves comuns ou fazer junções (join) de diferentes maneiras
    @classmethod
    def mergeDataFrames(cls, other_dataframe, how='inner', on=None):
        merged_data = cls.dataframe.merge(other_dataframe, how=how, on=on)
        return merged_data

    ### Adicione métodos para agregar dados usando funções como soma, média, contagem, etc., agrupando por colunas específicas
    @classmethod
    def aggregateData(cls, group_by_column, aggregation_function):
        aggregated_data = cls.dataframe.groupby(group_by_column).agg(aggregation_function)
        return aggregated_data

    ###  Se você estiver familiarizado com SQL, pode adicionar métodos que permitem realizar consultas SQL diretamente no DataFrame
    @classmethod
    def runSQLQuery(cls, sql_query):
        result = pd.read_sql_query(sql_query, cls.dataframe)
        return result

    ### Métodos para identificar e lidar com outliers nos dados, como substituição por valores médios ou remoção
    @classmethod
    def handleOutliers(cls, column, method='mean'):
        if method == 'mean':
            mean = cls.dataframe[column].mean()
            std_dev = cls.dataframe[column].std()
            lower_bound = mean - (2 * std_dev)
            upper_bound = mean + (2 * std_dev)
            cleaned_data = cls.dataframe[(cls.dataframe[column] >= lower_bound) & (cls.dataframe[column] <= upper_bound)]
        # Adicione mais métodos de tratamento de outliers conforme necessário

        return cleaned_data

        

# ex = PandasManager
# ex.importDataframeFromURL(url_file)
# print(ex.filtrarPor('modelo', 'Toro'))