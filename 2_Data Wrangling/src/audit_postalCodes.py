import pprint
import re
import codecs
import xml.etree.cElementTree as ET
import json
from collections import defaultdict
import os

pyPath = os.path.dirname(os.path.realpath(__file__))


OSMFILE = pyPath + "/Riyadh.osm"

bad_postal_codes = []
fixed_postal_codes = []

def audit_postal_code(postal_code):
    
    if len(postal_code) == 5 and postal_code.isdigit() :
        return postal_code


    bad_postal_codes.append(postal_code)

    # fix the code by taking only first 5 chars if it's more than 5 chars
    # this will not solve all problems but it'll reduce it.

    if len(postal_code) > 5 : 
        postal_code = postal_code[:5]
        fixed_postal_codes.append(postal_code)

    return postal_code



def is_postal_code(address_key):
    return address_key == 'addr:postcode'

def audit(osmfile):
    osm_file = open(osmfile, "r")
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_postal_code(tag.attrib['k']):
                    audit_postal_code(tag.attrib['v'])
    osm_file.close()
    return None


def test():
    audit(OSMFILE)
    print ('Bad postal code:', bad_postal_codes)
        
    print ('Bad postal code after fixing:', fixed_postal_codes)

if __name__ == '__main__':
    test()