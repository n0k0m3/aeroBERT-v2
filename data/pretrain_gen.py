import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from pretrain_xml import get_child_div,parse_ecfr_xml,section_regex
from lxml import etree
from datasets import load_dataset

if __name__ == "__main__":
    with open("./pretrain/title-14.xml", "r") as f:
        tree = etree.parse(f)
    root = tree.getroot()
    child_div = get_child_div(root)
    output_lines = parse_ecfr_xml(tree, child_div)
    full_text = "".join(output_lines)
    full_text = section_regex(full_text)

    with open("./pretrain/title-14.txt", "w") as f:
        f.write(full_text)

    data = load_dataset("text", data_files="./pretrain/title-14.txt")
    data = data["train"].train_test_split(test_size=0.2)
    train = [v["text"]+"\n" for v in data["train"].to_list()]
    test = [v["text"]+"\n" for v in data["test"].to_list()]
    with open("./pretrain/train.txt", "w") as f:
        f.writelines(train)
    with open("./pretrain/test.txt", "w") as f:
        f.writelines(test)