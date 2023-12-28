from lxml import etree
import itertools
import re

def check_ancestor(element, ancestor=["td", "th", "br", "table"]):
    if type(ancestor) == str:
        ancestor = [ancestor]
    while element is not None:
        if element.tag.lower() in ancestor:
            return True
        element = element.getparent()
    return False


def get_child_div(root):
    child_div = {}
    for element in root.iter("*"):
        if element.getparent() is None:
            continue
        ancestor_tag = element.getparent().tag
        _ = ancestor_tag.lower() in ["div" + str(i) for i in range(1, 10)]
        if _ and element.text != None and element.text.strip() != "":
            if ancestor_tag not in child_div:
                child_div[ancestor_tag] = []
            child_div[ancestor_tag].append(element.tag)
    child_div = {k: list(set(v)) for k, v in child_div.items()}
    child_div = list(set(itertools.chain.from_iterable(child_div.values())))
    # drop xref
    child_div = [tag for tag in child_div if tag.lower() != "xref"]
    return child_div


PROCESS_DICT = {
    "ac": {
        "b": ("after", "dot"),
        "8": ("after", "bar"),
    },
    "e": {
        "52": ("before", "_"),
        "7334": ("before", "_"),
        "9145": ("before", "_"),
        "54": ("before", "_"),
        "51": ("before", "^"),
        "7501": ("after", "bar"),
    },
    "su": ("before", "^"),
    "sup": ("before", "^"),
}


def process_text(element):
    if element.text is None or element.text.strip() == "":
        return ""
    text = element.text
    if element.tag.lower() in ["ac", "e"]:
        # try:
        op = None
        if PROCESS_DICT.get(element.tag.lower(), None):
            if PROCESS_DICT[element.tag.lower()].get(element.attrib["T"], None):
                op = PROCESS_DICT[element.tag.lower()][element.attrib["T"]][0]
        if op == "before":
            text = (
                PROCESS_DICT[element.tag.lower()][element.attrib["T"]][1]
                + text.lstrip()
            )
        elif op == "after":
            text = (
                text.rstrip()
                + PROCESS_DICT[element.tag.lower()][element.attrib["T"]][1]
            )
        else:
            pass
    elif element.tag.lower() in ["su", "sup"]:
        text = PROCESS_DICT[element.tag.lower()][1] + text.lstrip()
    if element.tail is not None and element.tail.strip() != "":
        text = text + element.tail
    return text


def parse_ecfr_xml(tree, child_div):
    output_lines = []
    for e_root in tree.xpath(" | ".join([f".//{tag}" for tag in child_div])):
        output_line = ""
        for e in e_root.iter():
            if check_ancestor(e):
                continue
            if e.tag.lower() == "table":
                continue
            output_line += process_text(e)
        output_lines.append(output_line.rstrip() + "\n")
    return output_lines


def section_regex(text):
    regex = r"§\s*([0-9]+)\.([0-9]+)*"
    text = re.sub(regex, r"§ \1-\2", text)
    text = text.replace("§§", "Sections")
    text = text.replace("§", "Section")
    return text


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    with open("./title-14.xml", "r") as f:
        tree = etree.parse(f)
    root = tree.getroot()
    child_div = get_child_div(root)
    output_lines = parse_ecfr_xml(tree, child_div)
    full_text = "".join(output_lines)
    full_text = section_regex(full_text)

    with open("./title-14.txt", "w") as f:
        f.write(full_text)
