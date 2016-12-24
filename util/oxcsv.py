import csv



def parse_ox_csv(filename):
    result = []
    with open(filename, "r") as f:
        reader = csv.reader(f)

        # pull of column names
        header = next(reader)
        for row in reader:
            result_row = []
            # pull off row names
            data = row[1:]
            for cell in data:
                try:
                    result_row.append(float(cell))
                except ValueError:
                    result_row.append(None)
            result.append(result_row)
    return result

if __name__ == "__main__":

    t1 = parse_ox_csv("../resources/testdata/1/raw_data.csv")
    print(t1[0])


