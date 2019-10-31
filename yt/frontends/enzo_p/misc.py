import numpy as np

from yt.funcs import ensure_tuple


def bdecode(block):
    """
    Decode a block descriptor to get its left and right sides and level.

    A block string consisting of (0, 1), with optionally one colon. The
    number of digits after the colon is the refinemenet level. The combined
    digits denote the binary representation of the left edge.
    """

    if ":" in block:
        level = len(block) - block.find(":") - 1
    else:
        level = 0
    bst = block.replace(":", "")
    d = float(2 ** len(bst))
    left = int(bst, 2)
    right = left + 1
    left /= d
    right /= d
    return level, left, right


def get_block_string_and_dim(block, min_dim=3):
    mybs = block[1:].split("_")
    dim = max(len(mybs), min_dim)
    return mybs, dim


def get_block_level(block):
    if ":" in block:
        l = block.find(":")
    else:
        l = len(block)
    return l


def get_block_info(block, min_dim=3):
    """Decode a block name to get its left and right sides and level.

    Given a block name, this function returns the locations of the block's left
    and right edges (measured as binary fractions of the domain along each
    axis) and level.

    Unrefined blocks in the root array (which can each hold an of octree) have
    a refinement level of 0 while their ancestors (used internally by Enzo-p's
    solvers - they don't actually hold meaningful data) have negative levels.
    Because identification of negative refinement levels requires knowledge of
    the root array shape (the 'root_blocks' value specified in the parameter
    file), all unrefined blocks are assumed to have a level of 0.
    """
    mybs, dim = get_block_string_and_dim(block, min_dim=min_dim)
    left = np.zeros(dim)
    right = np.ones(dim)
    level = 0
    for i, myb in enumerate(mybs):
        if myb == "":
            continue
        level, left[i], right[i] = bdecode(myb)
    return level, left, right


def get_root_blocks(block, min_dim=3):
    mybs, dim = get_block_string_and_dim(block, min_dim=min_dim)
    nb = np.ones(dim, dtype=int)
    for i, myb in enumerate(mybs):
        if myb == "":
            continue
        s = get_block_level(myb)
        nb[i] = 2 ** s
    return nb


def get_root_block_id(block, min_dim=3):
    mybs, dim = get_block_string_and_dim(block, min_dim=min_dim)
    rbid = np.zeros(dim, dtype=int)
    for i, myb in enumerate(mybs):
        if myb == "":
            continue
        s = get_block_level(myb)
        if s == 0:
            continue
        rbid[i] = int(myb[:s], 2)
    return rbid


def get_child_index(anc_id, desc_id):
    cid = ""
    for aind, dind in zip(anc_id.split("_"), desc_id.split("_")):
        cid += dind[len(aind)]
    cid = int(cid, 2)
    return cid


def is_parent(anc_block, desc_block):
    dim = anc_block.count("_") + 1
    if (len(desc_block.replace(":", "")) - len(anc_block.replace(":", ""))) / dim != 1:
        return False

    for aind, dind in zip(anc_block.split("_"), desc_block.split("_")):
        if not dind.startswith(aind):
            return False
    return True


def nested_dict_get(pdict, keys, default=None):
    """
    Retrieve a value from a nested dict using a tuple of keys.

    If a is a dict, and a['b'] = {'c': 'd'},
    then nested_dict_get(a, ('b', 'c')) returns 'd'.
    """

    keys = ensure_tuple(keys)
    val = pdict
    for key in keys:
        if val is None:
            break
        val = val.get(key)
    if val is None:
        val = default
    return val


def field_subgroup_generator(parameters, only_existing = False):
    """
    Create an iterator for the field subgroups in the parameter file.

    Within the parameter file, the Field parameter group can contain parameter
    subgroups named after the defined fields. These subgroups can contain
    details about a field's centering or its membership in field groups.

    The resulting generator iterator yields a 2-element tuple defined field.
    The first element indicates the field name while the second is a dict. If
    it was specified, the information in the subgroup can be found in the dict

    If only_existing is True, tuples are only be yielded for fields with
    associated specified subgroups. When False, tuples are yielded for all
    fields (empty dictionaries are used for fields without subgroups).
    """
    field_group = parameters.get("Field",[])
    for field in field_group.get('list',[]):
        subgroup = field_group.get(field, None)

        # accounts for the potential collision of having a "gamma" field and
        # assigning "gamma" a scalar value
        if (field == 'gamma') and (not isinstance(subgroup,dict)):
            subgroup = None

        if subgroup is None:
            if only_existing:
                continue #skip over this field
            else:
                subgroup = {}

        yield field,subgroup

def group_field_members(parameters, group_name):
    """
    Returns a set containing the members of the specified group.
    """
    # check within the Group parameter group.
    # Group : <group_name> : field_list
    out = set(nested_dict_get(group_name, (group_name, 'field_list'),
                              default = []))

    # iterate over field ubgroups and check the group_list parameter
    # Field : <field_name> : group_list
    for field, subgroup in field_subgroup_generator(parameters, True):
        if group_name in subgroup.get('group_list',[]):
            out.add(field)
    return out
