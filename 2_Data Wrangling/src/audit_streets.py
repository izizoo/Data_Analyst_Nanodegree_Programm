import pprint
import re
import codecs
import xml.etree.cElementTree as ET
import json
from collections import defaultdict
import os

pyPath = os.path.dirname(os.path.realpath(__file__))


street_types = re.compile(r'\b\S+\.?$', re.IGNORECASE)

OSMFILE = pyPath + "/Riyadh.osm"

fixed_street_names = []

expected = ["Street", "Road"]


street_mapping = { "St": "Street",
            "Rd" :"Road",
            "RD":"Road",
            "street":"Street"
}

def audit_street_type(street_name):
    """Check the street name """
    match = street_types.search(street_name)
    if match:
        street_type = match.group()
        if street_type not in expected and isEnglish(street_name):
            return update_street_name(street_name, street_mapping)


    return street_name


def update_street_name(name, mapping):
    """Fix the name and return or return it as it is if it's ok"""
    for key in mapping.keys():
        if re.search(key, name):
            name = re.sub(key, mapping[key], name)
            fixed_street_names.append(name)

    return name


def is_street_name(address_key):
    return address_key == 'addr:street'

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag.attrib['k']):
                    audit_street_type(tag.attrib['v'])
    osm_file.close()
    return street_types


def test():
    st_types = audit(OSMFILE)
    for st_type, ways in st_types.items():
        for name in ways:
            better_name = update_name(name, mapping)
            print ( name, "=>", better_name )
            if name == "Takhassusi St.":
                assert better_name == "Takhassusi Street"
            if name == "Omar Ibn Al Khattab Rd":
                assert better_name == "Omar Ibn Al Khattab Road"
    print(fixed_street_names)            

if __name__ == '__main__':
    test()
