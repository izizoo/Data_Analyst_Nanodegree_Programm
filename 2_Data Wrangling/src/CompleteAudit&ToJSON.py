import pprint
import re
import codecs
import xml.etree.cElementTree as ET
import json
from collections import defaultdict
import os

pyPath = os.path.dirname(os.path.realpath(__file__))



problem_chars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
street_types = re.compile(r'\b\S+\.?$', re.IGNORECASE)

OSMFILE = pyPath + "/Riyadh.osm"
CREATED = ["version", "changeset", "timestamp", "user", "uid"]


expected = ["Street", "Road"]


street_mapping = { "St": "Street",
            "Rd" :"Road",
            "RD":"Road",
            "street":"Street"
}

fixed_street_names = []
bad_postal_codes = []
fixed_postal_codes = []

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


def process_element(element):
    """
    Parse, validate and format node and way xml elements.
    Return list of dictionaries
    Keyword arguments:
    element -- element object from xml element tree iterparse
    """
    if element.tag == 'node' or element.tag == 'way':

        # Add empty tags - created (dictionary) and type (key/value )
        node = {'created': {}, 'type': element.tag}

        # Update pos array with lat and lon
        if 'lat' in element.attrib and 'lon' in element.attrib:
            node['pos'] = [float(element.attrib['lat']), float(element.attrib['lon'])]

        # Deal with node and way attributes
        for k in element.attrib:

            if k == 'lat' or k == 'lon':
                continue
            if k in CREATED:
                node['created'][k] = element.attrib[k]
            else:
                # Add everything else directly as key/value items of node and way
                node[k] = element.attrib[k]

        # Deal with second level tag items
        for tag in element.iter('tag'):
            k = tag.attrib['k']
            v = tag.attrib['v']

            # Search for problem characters in 'k' and ignore them
            if problem_chars.search(k):
                # Add to array to print out later
                continue
            elif k.startswith('addr:'):
                address = k.split(':')
                if len(address) == 2:
                    if 'address' not in node:
                        node['address'] = {}
                    if is_street_name(k):
                        v = audit_street_type(v)
                    if is_postal_code(k):
                        v = audit_postal_code(v)
                    node['address'][address[1]] = v
            else:
                node[k] = v

        # Add nd ref as key/value pair from way
        node_refs = []
        for nd in element.iter('nd'):
            node_refs.append(nd.attrib['ref'])

        # Only add node_refs array to node if exists
        if len(node_refs) > 0:
            node['node_refs'] = node_refs

        return node
    else:
        return None


def process(file_in, pretty=True):
    file_out = "{0}.json".format(file_in,"utf-8")

    data = []
    with codecs.open(file_out, "w", "utf-8") as fo:

        for _, element in ET.iterparse(file_in):
            el = process_element(element)
            if el:
                data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2,ensure_ascii=False) + "\n")
                else:
                    fo.write(json.dumps(el,ensure_ascii=False) + "\n")

        # Print summary
        print ('Fixed street names:', fixed_street_names)
        print ('Bad postal code:', bad_postal_codes)
        
        print ('Bad postal code after fixing:', fixed_postal_codes)


    return data


def test():

    data = process(OSMFILE, True)
    #You can print the data, just uncomment next line.
    #pprint.pprint(data)


if __name__ == "__main__":
    test()