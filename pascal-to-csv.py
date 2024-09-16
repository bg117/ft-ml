import glob, sys, getopt, os
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    train_xml = glob.glob(os.path.join(path + "train", "*.xml"))
    valid_xml = glob.glob(os.path.join(path + "valid", "*.xml"))
    test_xml = glob.glob(os.path.join(path + "test", "*.xml"))

    xml_list.extend(xml_parse(train_xml, "TRAIN"))
    xml_list.extend(xml_parse(valid_xml, "VALIDATION"))
    xml_list.extend(xml_parse(test_xml, "TEST"))

    return pd.DataFrame(xml_list)

def xml_parse(d, t):
    xml_list = []
    for xml_file in d:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for member in root.findall("object"):
            bbx = member.find("bndbox")
            xmin = int(bbx.find("xmin").text)
            ymin = int(bbx.find("ymin").text)
            xmax = int(bbx.find("xmax").text)
            ymax = int(bbx.find("ymax").text)
            label = member.find("name").text
            width = int(root.find("size")[0].text)
            height = int(root.find("size")[1].text)

            value = (
                t,
                xml_file,
                label,
                round(xmin / width, 2),
                round(ymin / height, 2),
                "",
                "",
                round(xmax / width, 2),
                round(ymax / height, 2),
                "",
                ""
            )

            xml_list.append(value)

    return xml_list

def main(argv):

    dir = None
    
    try:

        opts, _ = getopt.getopt(argv, "hd:", ["help","directory="])
        for opt, arg in opts:
            if opt == "-h, --help":
                raise Exception()
            elif opt in ("-d", "--directory"):
                dir = str(arg)
        
        if (dir is None):
            raise Exception()
    
    except Exception:
        print("Specify a directory that contains XML files.")
        print("pascal-to-csv.py -d <directory>")
        sys.exit(2)

    xml_df = xml_to_csv(dir)
    xml_df.to_csv(dir + "/dataset.csv", index=None, header=None)


if __name__ == "__main__":
   main(sys.argv[1:])
