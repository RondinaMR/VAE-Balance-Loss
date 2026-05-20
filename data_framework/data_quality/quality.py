import pandas as pd
from pandas.errors import ParserError
from collections import Counter
import numpy as np
import urllib
import urllib.request
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import sys
from tqdm.auto import tqdm

n_of_decimal = 3


# Pre Processing

def open_file(a, isurl, symbol=None):
    try:
        # return if the files is correct, so it can be opened, and return the delimiter of the csv.
        correct_file, symbol = detect_symbol(a, isurl, symbol)
        print(f'symbol \'{symbol}\'')
        if correct_file:
            print("file is correct")
            df = pd.read_csv(a, delimiter=symbol, skip_blank_lines=False, on_bad_lines='warn', index_col=False)
        else:
            return False, symbol, "", symbol  # as shortcut: if the file is wrong, the second parameter "symbol" now corresponds to the error that was raised
    except UnicodeDecodeError as chracter_error:
        df = pd.read_csv(a, delimiter=symbol, skip_blank_lines=False, encoding='latin1',
                         on_bad_lines='warn')  # I change the encoding in case of UnicodeEncodeError
        print("Program wil MAY stop due to the error ",
              chracter_error)  # The program may stop anyway; it is rare but it can happen
    except HTTPError as net_error:
        try:
            # we can try to change the user agent; this is helpful for network error 403 Forbidden
            req = Request(a)
            req.add_header('User-Agent',
                           'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36')
            content = urlopen(req)
            df = pd.read_csv(content, delimiter=symbol, skip_blank_lines=False, on_bad_lines='warn', index_col=False)
        except:
            print("Program wil stop due to the error ", net_error)
            return False, "Network error", "", net_error
    except FileNotFoundError as file_error:
        print("Program wil stop due to the error ", file_error)
        return False, "File is not found", "", file_error
    except ParserError as architecture_error:
        print("Program wil stop due to the error ", architecture_error)
        return False, "Architecture error", "", architecture_error
    except TypeError as type_error:
        print("FIle is 0 byte")
        return False, "File is 0 byte", "", type_error
    except:
        # unpredictable error
        print("Fatal error")
        print("Unexpected error:", sys.exc_info()[0])
        return False, "There is something wrong", "", sys.exc_info()[0]
    if df.shape[0] == 0:
        return False, "No rows", "", "The table has no rows"
    print(df.iloc[0])  # print the first row
    df = df.replace(r'^\s*$', np.NaN, regex=True)  # <-- Replace the " " as NaN
    print()
    return True, df, symbol, "No error"


def detect_symbol(file, isurl, symbol=None):
    ext = file[-4:].lower().strip()
    print(ext)
    if ext == "zip":
        message = "This file is a zip"
        print(message)
        return False, message
    if isurl:
        try:
            content = urllib.request.urlopen(file)
        except:
            try:
                # if I have an error, I can try to change the user agent; helpful in case of Error 403 Forbidden
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36'}
                req = urllib.request.Request(url=file, headers=headers)
                content = urllib.request.urlopen(req)  # .read()
            except:
                message = "fatal error during detection symbol, can't open the file (url)"
                print(message)
                return False, message
    else:
        try:
            content = open(file, "r")
        except:
            content.close()
            message = "fatal error during detection symbol, can't open the file (no url)"
            print(message)
            return False, message
    not_done = True
    detected_symbol = ""
    for line in content:
        if (not_done):
            try:
                if isurl:
                    decoded_line = line.decode("utf-8")
                    line = decoded_line[0:]
                if symbol is not None:
                    return True, symbol
                detected_symbol = return_symbol(line)  # this is where I find the delimiter of the CSV
                not_done = False
                content.close()
                if (line.startswith("{")):
                    message = "The file stars with a {, probably it is a json message"
                    print(message)
                    return False, message
                else:
                    return True, detected_symbol
            except:
                # If I have an error on the first row, I try the next one
                pass
    content.close()


def return_symbol(line):
    # return the character that maximixe the number of split. Line is, hopefully and in most cases, the header.
    n_semicolon = len(line.split(";"))
    n_comma = len(line.split(","))
    n_tab = len(line.split("\t"))
    n_spaces = len(line.split(" "))
    n_onecolons = len(line.split(":"))
    n_twocolons = len(line.split("::"))
    n_max = max([n_semicolon, n_comma, n_tab, n_spaces, n_onecolons, n_twocolons])
    symbol = ""
    if n_semicolon == n_max:
        symbol = ";"
    elif n_comma == n_max:
        symbol = ","
    elif n_tab == n_max:
        symbol = "\t"
    elif n_spaces == n_max:
        symbol = " "
    elif n_onecolons == n_max:
        symbol = ":"
    elif n_twocolons == n_max:
        symbol = "::"
    if symbol == "":
        raise ValueError("Unrecognized symbol")
    return symbol


def extract_basic_info(df):
    # extract n rows, n columns, name columns
    n = df.shape
    n_rows = n[0]
    n_columns = n[1]
    columns = list(df.columns)
    print("N rows:", n_rows, "\nN columns:", n_columns, "\nCollumns: ", columns[:])
    print()
    return (n_rows, n_columns, columns)


def extract_metadata(df):
    # extract types of columns, number of non null values for each column
    columns_type = list(df.dtypes)
    print("Columns_type ", columns_type)
    n_non_null_values = list(df.count())
    print("Non null values for each column ", n_non_null_values)
    print()
    return (columns_type, n_non_null_values)


# Measures

## ISO

# Completeness measures----------------------------------------------------->

def empy_records(df, n_rows):
    # ISO: Com-I-5 Correct
    i = 0  # counter for the number of rows full of NaN
    for el in df.isnull().all(1).values:
        if (el):
            i += 1
    ratio = round(1 - i / n_rows, n_of_decimal)
    print("ISO: Com-I-5: empty records ratio:", ratio)
    return ratio


# Accuracy measures----------------------------------------------------------------->

def risk_of_dataset_inaccuracy(df, n_columns, n_rows):
    # ISO: Acc-I-4
    outliers_list = []
    counted_data_element = 0
    for i in range(0, n_columns):
        value_list = df.iloc[:, i].tolist()
        cleaned_list = []
        for x in value_list:
            val = str(x)
            if (val != 'nan'):  # we don't consider nan value
                new_val = val.replace(",", ".")
                try:
                    float_val = float(new_val)  # If error, the data item is not a number
                    cleaned_list.append(float_val)
                except:
                    pass
        if (len(cleaned_list) > 4):  # we need at least 5 elements to talk about outliers
            # the IQR method is used to detect outliers
            labels = []
            arr_x = np.array(cleaned_list)
            q1 = np.percentile(arr_x, 25)
            q3 = np.percentile(arr_x, 75)
            IQR = q3 - q1
            min = q1 - 1.5 * IQR
            max = q3 + 1.5 * IQR
            for el in arr_x:
                if (el > max or el < min):
                    labels.append(-1)
                else:
                    labels.append(0)
            outlier = -1
            n_outlier = labels.count(outlier)
            if (n_outlier == len(cleaned_list)):
                # All the numbers are outliers!
                outliers_list.append(0)
            else:
                # optimal case, there are true outliers
                counted_data_element += len(cleaned_list)
                outliers_list.append(n_outlier)
        else:
            outliers_list.append(0)
    print("Number of outliers for numeric columns", outliers_list)
    counted_outliers = sum(outliers_list)
    if (counted_data_element) > 0:
        print("Counted outliers: ", counted_outliers, " Counted data elemnts: ", counted_data_element)
        ratio = round(counted_outliers / counted_data_element, n_of_decimal)
    else:
        print("Can't compute because there are no int or float elements")
        ratio = np.NaN
    print("ISO ACC-I-4: risk_of_dataset_inaccuracy:", ratio)
    return (ratio)


# Consistency measures------------------------------------------------------------------>

# Con-I-3
def risk_of_data_inconsistency(df, n_columns):
    total_elements = 0
    field_array = []
    single_duplication = []
    n_rows = len(df)
    for i in range(0, n_columns):
        print("column: ", i, end='; ')
        value_list = df.iloc[:, i].tolist()  # [a,b,c]
        total_elements += len(value_list)
        field_array.append(value_list)  # I need this for the second computation --> [[value_list1],[value_list2],..]
        duplication_sum = sum(return_n_duplicates(value_list))
        single_duplication.append(duplication_sum)
    total_possible_duplications = n_rows * (n_columns + (n_columns * (n_columns - 1) / 2)) if n_columns > 1 else n_rows * n_columns
    print("\ntotal elements: ", total_elements)
    print(f"\ntotal possible duplications {total_possible_duplications} ({n_rows} rows, {n_columns} columns)")
    print("single duplication: ", single_duplication)
    total_single_duplication = sum(single_duplication)
    print("total_single_duplication: ", total_single_duplication)
    total_multiple_duplication = compute_duplication_multiple_columns(field_array)
    print("total_multiple_duplication: ", total_multiple_duplication)
    final_total = total_multiple_duplication + total_single_duplication
    ratio = round(final_total / total_possible_duplications, n_of_decimal)
    print("ISO CON-I-3: risk_of_data_inconsistency:", ratio)
    return ratio


def return_n_duplicates(a):
    a_set = set(a)
    contains_duplicates = [a.count(element) for element in
                           tqdm(a_set,
                                desc=f'Counting duplicates',delay=3)]  # --> how many times each single value is present in the list a --> example: [1,3,2,1,4,1,2]
    duplicates = [x for x in contains_duplicates if
                  x > 1]  # example: [3,2,4,1] so we don't consider the value that occurs only 1
    return duplicates


def compute_duplication_multiple_columns(big_array):
    total_multiple_duplication = 0
    for i in range(0, len(big_array) - 1):
        for j in range(i + 1, len(big_array)):
            first_array = big_array[i]
            second_array = big_array[j]
            union_array = []
            for a in range(0, len(big_array)):
                try:
                    new_el = str(first_array[a]) + str(second_array[a])
                    union_array.append(new_el)
                except:
                    pass
            duplicates = sum(return_n_duplicates(union_array))
            total_multiple_duplication += duplicates
    return total_multiple_duplication


## Not ISO

def empty_cells(df, n_rows, n_col):
    n_notnan = df.notna().sum().sum()
    ratio = round(n_notnan / (n_rows * n_col), n_of_decimal)
    print("\nISO Related: ratio of empty cells: ", ratio)
    print()
    return ratio


def risk_type_inconsistency(df, n_col):
    n_no_consistency = 0
    n_data_evaluated = 0
    for i in range(0, n_col):
        # for each column
        value_list = df.iloc[:, i].tolist()
        type_list = [str(x).replace('.', '').replace(',', '').isdigit() for x in value_list if str(x) != 'nan']
        # Number as "4,4" or "4.2" create only confusion for this measure; so we don't consider the "," and ".". So any number becomes integer.
        # The data type is integer or string
        n_data_evaluated += len(type_list)
        # If there is a NaN i don't consider it
        # I take the string, i replace the . and I check if it is composed by only digit; if yes, it is a float
        type_set = set(type_list)
        if (len(type_set) > 1):
            print("2 Types!!! Error in column", i)
            type_counter = list(Counter(type_list).values())
            print(type_counter)
            min_element = min(type_counter)
            n_no_consistency += min_element  # the type that repreents the minority of the elements is the error
    ratio = 1-round(n_no_consistency / n_data_evaluated, n_of_decimal)
    print("\nISO Related: Risk of type inconsistency", ratio)
    return ratio


def replace_sub_string(line, arr):
    # replace in the whole are of the sub_string with an "a"
    # example: I, Am, "Giorno,Giovanna" --> I, Am, "a"
    # in this way, i don't consider the separator inside the quotes
    new_line = ""
    # arr is the array containing the indexes of the quotes
    # line is the original record
    for i in range(0, len(arr)):
        if (i == 0):
            new_line += line[:arr[i]] + " a "
            # the original record is taken until the first quote
        elif (i == (len(arr) - 1)):
            new_line += line[arr[i]:]
            # the original record is taken from the last quote until the end
        elif i % 2 != 0:
            new_line += line[arr[i]:arr[i + 1]] + " a "
            # when i is uneven, we ignore the line[arr[i]:arr[i+1]] because
            # it contains the record within the quotes
            # Instead, if i is even, line[arr[i]:arr[i+1]] is outside of the quotes
            # so we can take it
    return new_line


def return_cleaned_line(line):
    double = []
    for i in range(0, len(line)):
        if line[i] == "\"":  # try to find the character "
            double.append(i)
    if len(double) == 0:  # there are no " chracters
        return line  # return the line as it is
    else:
        return replace_sub_string(line, double)  # symbol = "\""


def compute_arch_cons(ur, symbol, isurl):
    # Con-I-4
    if isurl:
        try:
            content = urllib.request.urlopen(ur)
        except:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36'}
            req = urllib.request.Request(url=ur, headers=headers)
            content = urllib.request.urlopen(req)
    else:
        try:
            content = open(ur, "r")
            detect_encoding(content)
        except UnicodeDecodeError:
            # Try latin1 enconding
            content.close()
            content = open(ur, "r", encoding="latin1")
        except:
            message = "error during compute_arch_cons, can't open the file"
            print(message)
            return False, message
    n_list = []
    i = -1  # starts from -1 because we want to consider the first row that is the header
    n_rows_smaller = 0
    n_rows_bigger = 0

    for line in content:
        i += 1  # total rows in the original file
        if isurl == True:
            try:
                decoded_line = line.decode("utf-8")
                line = decoded_line[0:]
            except:
                decoded_line = line.decode("latin1")
                line = decoded_line[0:]
        # eliminate all the " ... " substring part. Because A,B,"C,D" contains 3 cells, no 4.
        new_line = return_cleaned_line(line)
        n = len(new_line.split(symbol))
        if (i == 0):  # this is the header
            n_columns = n  # I take its number of cells as parameter
        if (n < n_columns):
            n_rows_smaller += 1
        elif (n > n_columns):
            n_rows_bigger += 1
    content.close()
    print("Total Row: ", i, "N rows that exceed the number of cells: ", n_rows_bigger,
          "N rows whose n of cells is lower: ", n_rows_smaller)
    ratio = round((i - n_rows_bigger - n_rows_smaller) / i, n_of_decimal)
    print("\nISO ISO Related: architecture consistency ", ratio)
    return ratio


# Single file handler
# def file_handler(input_file, isurl, symbol=None):
#     # return measures of a single file
#     correct_file, dff, symbol, err = open_file(input_file, isurl, symbol)
#     if correct_file:
#         # pre-processing
#         print("\n---PRE-PROCESSING---\n")
#         n_rows, n_columns, columns = extract_basic_info(dff)
#         columns_type, n_non_null_values = extract_metadata(dff)
#         print("\n---COMPLETENESS---\n")
#         # Completeness measures
#         ratio_empty_cells = empty_cells(dff, n_rows, n_columns)
#         ratio_empty_records = empy_records(dff, n_rows)
#         print("\n---ACCURACY---\n")
#         # Accuracy measures
#         ratio_risk_of_dataset_inaccuracy = risk_of_dataset_inaccuracy(dff, n_columns, n_rows)
#         print("\n---CONSISTENCY---\n")
#         # Connsistency measures
#         print("Consistency measures: Risk of data inconsistency")
#         ratio_data_inconsistency = risk_of_data_inconsistency(dff, n_columns)
#         # Iso-derivated measures
#         print("Iso-derivated measures")
#         print("Risk of type inconsistency")
#         risk_type_insistency = risk_type_inconsistency(dff, n_columns)
#         print("Architecture consistency")
#         architecture_consistency = compute_arch_cons(input_file, symbol, isurl)
#         # all measures
#         all_measures = [1, ratio_empty_cells, ratio_empty_records, ratio_risk_of_dataset_inaccuracy,
#                         ratio_data_inconsistency, risk_type_insistency, architecture_consistency, err]
#     else:
#         print("Error --> ", dff)
#         all_measures = [0, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, err]
#     return all_measures


def detect_encoding(fileobj):
    """Checks the encoding of a file opened with 'open()'."""
    # Read the first three bytes from the file
    fileobj.seek(0)
    byte_str = fileobj.read(3)
    fileobj.seek(0) #reset the read position

    # Check if the first two bytes match the expected encoding
    if len(byte_str) == 3:
        expected_bytes = b'\xef\xbb\xbf'
        if byte_str[0] != expected_bytes[0]:
            return None

        # If we reach here, the encoding seems valid
        return 'utf-8'
    else:
        raise UnicodeDecodeError("Encoding different from utf-8")


class Quality:
    # TODO add verbose=False
    # TODO add info logging
    def __init__(self, dataset_name, source_file_name, output_path, isurl, symbol=None, pretty_name=None):
        self._dataset_name = dataset_name
        if pretty_name is None:
            self._pretty_name = dataset_name
        else:
            self._pretty_name = pretty_name
        self._file_name = source_file_name  # file path
        self._symbol = symbol
        # file_name = input("Enter data file name: ")
        # label = input("Is there the header? [y/n]")

        # with open(self._file_name) as f:
        #     if self._header == "y":
        #         output_filename = f.readline() + ".csv"
        #     else:
        #         output_filename = input("Enter output name: ") + ".csv"
        output_filename = output_path + dataset_name + ".csv"
        textfile = open(output_filename, "w")
        # this write the columns name
        textfile.write("Dataset-Name,Can-Open,Com-I-1-DevA,Com-I-5,Acc-I-4,Con-I-3-DevC,Con-I-2-DevB,Con-I-4-DevD,Error\n")

        print("\nurl: ", self._file_name)

        # measures = file_handler(source_file_name, isurl, self._symbol)
        # return measures of a single file
        correct_file, dff, symbol, err = open_file(self._file_name, isurl, self._symbol)
        if correct_file:
            # pre-processing
            print("\n---PRE-PROCESSING---\n")
            n_rows, n_columns, columns = extract_basic_info(dff)
            columns_type, n_non_null_values = extract_metadata(dff)
            print("\n---COMPLETENESS---\n")
            # Completeness measures
            ratio_empty_cells = empty_cells(dff, n_rows, n_columns) #Com-I-1-DevA
            ratio_empty_records = empy_records(dff, n_rows) #Com-I-5
            print("\n---ACCURACY---\n")
            # Accuracy measures
            ratio_risk_of_dataset_inaccuracy = risk_of_dataset_inaccuracy(dff, n_columns, n_rows) #Acc-I-4
            print("\n---CONSISTENCY---\n")
            # Connsistency measures
            print("Consistency measures: Risk of data inconsistency")
            ratio_data_inconsistency = risk_of_data_inconsistency(dff, n_columns) #Con-I-3
            # Iso-derivated measures
            print("Iso-derivated measures")
            print("Risk of type inconsistency")
            risk_type_insistency = risk_type_inconsistency(dff, n_columns) #Con-I-2-DevB
            print("Architecture consistency")
            architecture_consistency = compute_arch_cons(self._file_name, symbol, isurl) #Con-I-4-DevC
            # all measures
            measures = [1, ratio_empty_cells, ratio_empty_records, ratio_risk_of_dataset_inaccuracy,
                        ratio_data_inconsistency, risk_type_insistency, architecture_consistency, err]
        else:
            print("Error --> ", dff)
            measures = [0, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, err]
        textfile.write(self._pretty_name)  # the [0;-1] is to get rid of the \n character
        for element in measures:
            textfile.write("," + str(element))
        textfile.write("\n")
        print("\n")
        textfile.close()
