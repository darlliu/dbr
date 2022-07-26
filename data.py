import random
import sqlite3
con = sqlite3.connect('D:/data/DanbooruDataset/danbooru.sqlite')
cur = con.cursor()


def _tokenize(text):
    """further tokenize the tags a little bit, not sure if needed for spe tokenizer"""
    return text.replace('_', ' ')


def _get_overused_tags():
    cur.execute(
        """select tag_string_general from posts limit 10000""")
    _all_tags = {}
    for r in cur.fetchall():
        for tag in r[0].split():
            tag = _tokenize(tag)
            if tag in _all_tags:
                _all_tags[tag] += 1
            else:
                _all_tags[tag] = 1
    overused_tags = sorted(_all_tags.items(), key=lambda x: -x[1])[:40]
    overused_tags = set([t[0] for t in overused_tags])

    def _get_overused():
        return overused_tags
    return _get_overused


def _get_char_names():
    cur.execute(
        """select tag_string_character from posts limit 10000""")
    _all_tags = {}
    for r in cur.fetchall():
        for tag in r[0].split():
            tag = _tokenize(tag)
            if tag in _all_tags:
                _all_tags[tag] += 1
            else:
                _all_tags[tag] = 1
    _all_names = [n for n in _all_tags.keys()]

    def _get():
        return _all_names
    return _get


def _get_max_ids():
    cur.execute(
        """select id from posts where length(tag_string_character) > 1""")
    _all_ids = []
    for r in cur.fetchall():
        _all_ids.append(int(r[0]))

    def _get_all():
        return _all_ids

    return _get_all


get_max_ids = _get_max_ids()
get_overused_tags = _get_overused_tags()
get_char_names = _get_char_names()


def _generate_train_test(split=0.2):
    full_ids = get_max_ids()
    random.shuffle(full_ids)
    split_at = int(max(get_max_ids()) * split)
    return full_ids[split_at:], full_ids[:split_at]


def _clean(ss):
    tokens = ss.split()
    return " ".join([t for t in tokens if t not in get_overused_tags()])


def get_data(id):
    """get relevant data from id, returns general tag, char tag, series tag and md5"""
    sql = f"""
    select tag_string_general, tag_string_character, tag_string_copyright, md5 from posts where id = {id} limit 1
    """
    rr = cur.execute(sql).fetchall()
    if len(rr):
        return _clean(_tokenize(rr[0][0])), _tokenize(rr[0][1]), _tokenize(rr[0][2]), rr[0][3]
    else:
        return None, None, None, None


def get_batch(ds, batch_size=16):
    cur_idx = 0
    while len(ds) > batch_size:
        batch, ds = ds[:batch_size], ds[batch_size:]
        batch = [get_data(ii) for ii in batch]
        yield [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch], [b[3] for b in batch]


TRAIN, TEST = _generate_train_test()

if __name__ == "__main__":
    print(get_overused_tags())
    print(get_char_names()[:100])
    for b in get_batch(TRAIN, 5):
        print(b)
        break
