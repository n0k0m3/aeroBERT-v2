from lxml import etree

with open("title-14.xml", "r") as f:
    tree = etree.parse(f)

root = tree.getroot()
output = []
for element in root.iter("*"):
    text = element.text or ""
    tail = element.tail or ""
    line = text.strip() + " " + tail.strip()
    if line == "":
        continue
    output.append(line + "\n")

with open("title-14.txt", "w+") as f:
    f.writelines(output)
