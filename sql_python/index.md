```python
import mysql.connector
import pymysql as mysql
from sqlalchemy import create_engine
import pandas as pd
```


```python
login = 'root'
passwd = '_Rootedroot7925'
server = 'localhost'
conn_string = 'mysql+mysqlconnector://{}:{}@{}'.format(login, passwd, server)
engine = create_engine(conn_string, echo=False, encoding='utf-8')
engine.execute('use classicmodels')
engine.execute('show tables').fetchall()
```




    [('customers',),
     ('employees',),
     ('offices',),
     ('orderdetails',),
     ('orders',),
     ('payments',),
     ('productlines',),
     ('products',),
     ('sales',)]




```python
create_table = """
                CREATE TABLE employee_sales(
                    sales_employee VARCHAR(50) NOT NULL,
                    fiscal_year INT NOT NULL,
                    sale DECIMAL(14,2) NOT NULL,
                    PRIMARY KEY(sales_employee,fiscal_year))
                """
engine.execute(create_table)
```




    <sqlalchemy.engine.cursor.LegacyCursorResult at 0x249399740d0>




```python
insert_into = """
                INSERT INTO employee_sales(sales_employee,fiscal_year,sale)
                VALUES
                    ('Bob',2016,100),
                    ('Bob',2017,150),
                    ('Bob',2018,200),
                    ('Alice',2016,150),
                    ('Alice',2017,100),
                    ('Alice',2018,200),
                    ('John',2016,200),
                    ('John',2017,150),
                    ('John',2018,250)
                """
engine.execute(insert_into)
```




    <sqlalchemy.engine.cursor.LegacyCursorResult at 0x24939f176d0>




```python
engine.execute('use classicmodels')
engine.execute('SHOW COLUMNS FROM employee_sales').fetchall()
```




    [('sales_employee', b'varchar(50)', 'NO', 'PRI', None, ''),
     ('fiscal_year', b'int', 'NO', 'PRI', None, ''),
     ('sale', b'decimal(14,2)', 'NO', '', None, '')]




```python
sales_fiscal = """
                SELECT 
                    fiscal_year, 
                    sales_employee,
                    sale,
                    NTILE (2) OVER (ORDER BY fiscal_year) total_sales
                FROM
                    employee_sales
                """
fiscal_sales = pd.read_sql(sales_fiscal, engine)
fiscal_sales
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fiscal_year</th>
      <th>sales_employee</th>
      <th>sale</th>
      <th>total_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016</td>
      <td>Alice</td>
      <td>150.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016</td>
      <td>Bob</td>
      <td>100.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>John</td>
      <td>200.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>Alice</td>
      <td>100.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>Bob</td>
      <td>150.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2017</td>
      <td>John</td>
      <td>150.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2018</td>
      <td>Alice</td>
      <td>200.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2018</td>
      <td>Bob</td>
      <td>200.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018</td>
      <td>John</td>
      <td>250.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create temp table in first SELECT. select from temp table with second SELECT.

query = """
        WITH productline_sales AS (
        SELECT 
            productline,
            year(orderDate) order_year,
            ROUND(SUM(quantityOrdered * priceEach),0) order_value
        FROM orders
        INNER JOIN orderdetails USING (orderNumber)
        INNER JOIN products USING (productCode)
        GROUP BY productline, order_year 
        )

        SELECT
            productline, 
            order_year, 
            order_value,
            NTILE(3) OVER (
                PARTITION BY order_year
                ORDER BY order_value DESC
            ) product_line_group
        FROM 
            productline_sales;
        """
n_tile = pd.read_sql(query, engine)
n_tile
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>productline</th>
      <th>order_year</th>
      <th>order_value</th>
      <th>product_line_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Classic Cars</td>
      <td>2003</td>
      <td>1374832.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Vintage Cars</td>
      <td>2003</td>
      <td>619161.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trucks and Buses</td>
      <td>2003</td>
      <td>376657.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Motorcycles</td>
      <td>2003</td>
      <td>348909.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Planes</td>
      <td>2003</td>
      <td>309784.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ships</td>
      <td>2003</td>
      <td>222182.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Trains</td>
      <td>2003</td>
      <td>65822.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Classic Cars</td>
      <td>2004</td>
      <td>1763137.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Vintage Cars</td>
      <td>2004</td>
      <td>854552.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Motorcycles</td>
      <td>2004</td>
      <td>527244.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Planes</td>
      <td>2004</td>
      <td>471971.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Trucks and Buses</td>
      <td>2004</td>
      <td>465390.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Ships</td>
      <td>2004</td>
      <td>337326.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Trains</td>
      <td>2004</td>
      <td>96286.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Classic Cars</td>
      <td>2005</td>
      <td>715954.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Vintage Cars</td>
      <td>2005</td>
      <td>323846.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Motorcycles</td>
      <td>2005</td>
      <td>245273.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Trucks and Buses</td>
      <td>2005</td>
      <td>182066.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Planes</td>
      <td>2005</td>
      <td>172882.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Ships</td>
      <td>2005</td>
      <td>104490.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Trains</td>
      <td>2005</td>
      <td>26425.0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
query = """
        CREATE TABLE productLineSales
        SELECT
            productLine,
            YEAR(orderDate) orderYear,
            quantityOrdered * priceEach orderValue
        FROM
            orderDetails
                INNER JOIN
            orders USING (orderNumber)
                INNER JOIN
            products USING (productCode)
        GROUP BY
            productLine ,
            YEAR(orderDate)
        """
engine.execute(query)
```




    <sqlalchemy.engine.cursor.LegacyCursorResult at 0x2493a0a30d0>




```python
pd.read_sql('SELECT * FROM productLineSales', engine)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>productLine</th>
      <th>orderYear</th>
      <th>orderValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Vintage Cars</td>
      <td>2003</td>
      <td>4080.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Classic Cars</td>
      <td>2003</td>
      <td>5571.80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trucks and Buses</td>
      <td>2003</td>
      <td>3284.28</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trains</td>
      <td>2003</td>
      <td>2770.95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ships</td>
      <td>2003</td>
      <td>5072.71</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Planes</td>
      <td>2003</td>
      <td>4825.44</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Motorcycles</td>
      <td>2003</td>
      <td>2440.50</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Classic Cars</td>
      <td>2004</td>
      <td>8124.98</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Vintage Cars</td>
      <td>2004</td>
      <td>2819.28</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Trains</td>
      <td>2004</td>
      <td>4646.88</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Ships</td>
      <td>2004</td>
      <td>4301.15</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Planes</td>
      <td>2004</td>
      <td>2857.35</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Motorcycles</td>
      <td>2004</td>
      <td>2598.77</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Trucks and Buses</td>
      <td>2004</td>
      <td>4615.64</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Motorcycles</td>
      <td>2005</td>
      <td>4004.88</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Classic Cars</td>
      <td>2005</td>
      <td>5971.35</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Vintage Cars</td>
      <td>2005</td>
      <td>5346.50</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Trucks and Buses</td>
      <td>2005</td>
      <td>6295.03</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Trains</td>
      <td>2005</td>
      <td>1603.20</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Ships</td>
      <td>2005</td>
      <td>3774.00</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Planes</td>
      <td>2005</td>
      <td>4018.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
#create temp table first.
query = """
        WITH temp AS (
            SELECT
                productLine,
                SUM(orderValue) orderValue
            FROM
                productLineSales
            GROUP BY
                productLine
        )
        
        SELECT
            productLine,
            orderValue,
            ROUND(
               PERCENT_RANK() OVER (
                  ORDER BY orderValue
               ), 2) percentile_rank
        FROM temp
        """
pd.read_sql(query, engine)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>productLine</th>
      <th>orderValue</th>
      <th>percentile_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trains</td>
      <td>9021.03</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Motorcycles</td>
      <td>9044.15</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Planes</td>
      <td>11700.79</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Vintage Cars</td>
      <td>12245.78</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ships</td>
      <td>13147.86</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Trucks and Buses</td>
      <td>14194.95</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Classic Cars</td>
      <td>19668.13</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
query = """
        SELECT
            productLine,
            orderYear,
            orderValue,
            ROUND(
            PERCENT_RANK()
            OVER (
                PARTITION BY orderYear
                ORDER BY orderValue
            ),2) percentile_rank
        FROM
            productLineSales
        """
pd.read_sql(query, engine)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>productLine</th>
      <th>orderYear</th>
      <th>orderValue</th>
      <th>percentile_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Motorcycles</td>
      <td>2003</td>
      <td>2440.50</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Trains</td>
      <td>2003</td>
      <td>2770.95</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trucks and Buses</td>
      <td>2003</td>
      <td>3284.28</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Vintage Cars</td>
      <td>2003</td>
      <td>4080.00</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Planes</td>
      <td>2003</td>
      <td>4825.44</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ships</td>
      <td>2003</td>
      <td>5072.71</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Classic Cars</td>
      <td>2003</td>
      <td>5571.80</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Motorcycles</td>
      <td>2004</td>
      <td>2598.77</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Vintage Cars</td>
      <td>2004</td>
      <td>2819.28</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Planes</td>
      <td>2004</td>
      <td>2857.35</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Ships</td>
      <td>2004</td>
      <td>4301.15</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Trucks and Buses</td>
      <td>2004</td>
      <td>4615.64</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Trains</td>
      <td>2004</td>
      <td>4646.88</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Classic Cars</td>
      <td>2004</td>
      <td>8124.98</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Trains</td>
      <td>2005</td>
      <td>1603.20</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Ships</td>
      <td>2005</td>
      <td>3774.00</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Motorcycles</td>
      <td>2005</td>
      <td>4004.88</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Planes</td>
      <td>2005</td>
      <td>4018.00</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Vintage Cars</td>
      <td>2005</td>
      <td>5346.50</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Classic Cars</td>
      <td>2005</td>
      <td>5971.35</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Trucks and Buses</td>
      <td>2005</td>
      <td>6295.03</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>


