import re
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from ..input_entity import ArcEntity, PlaceEntity, PetriNetFile
from ..output_entity import Move


class PntTranslator:
    _instance = None

    def __init__(self):
        self.petri_net_file = None
        self.curr_line_idx = 0
        self.lines = []

    @classmethod
    def get_pnt_translator(cls):
        if cls._instance is None:
            cls._instance = PntTranslator()
        return cls._instance

    def get_petri_net_file(self):
        return self.petri_net_file

    def translate_to_petri_net_file(self, path):
        self.curr_line_idx = 0
        self.petri_net_file = PetriNetFile()
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                self.lines = [line.rstrip("\n") for line in f.readlines()]
        except OSError:
            self.lines = []
        if not self.lines:
            return
        while self.curr_line_idx < len(self.lines):
            while self.curr_line_idx < len(self.lines) and (re.match(r"\s*$", self.lines[self.curr_line_idx]) or re.match(r"\s*//.*", self.lines[self.curr_line_idx])):
                self.curr_line_idx += 1
            if self.curr_line_idx == len(self.lines):
                break
            line = self.lines[self.curr_line_idx]
            if self._is_ef(line):
                self.petri_net_file.EFline = line
                self.curr_line_idx += 1
                continue
            if re.match(r"\s*P\s*M\s*PRE\s*,\s*POST\s*", line):
                self._deal_with_net_struct()
            elif self._is_value(line):
                field_name = self._get_field_name(line)
                self.petri_net_file.value_info[field_name] = self._make_value()
            elif self._is_map(self._get_data(line)):
                field_name = self._get_field_name(line)
                self.petri_net_file.map_info[field_name] = self._make_map()
            else:
                field_name = self._get_field_name(line)
                self.petri_net_file.set_info[field_name] = self._make_set()

    def _get_field_name(self, line):
        pattern = r".*=" if self._is_value(line) else r".*:"
        matcher = re.match(pattern, line)
        field_name = matcher.group(0)[:-1] if matcher else line
        return field_name.replace(" ", "")

    def _deal_with_net_struct(self):
        self.curr_line_idx += 1
        start_marking = {}
        net_struct = []
        while self.curr_line_idx != len(self.lines):
            place_info = self.lines[self.curr_line_idx]
            self.curr_line_idx += 1
            if re.match(r"\s*$", place_info) or re.match(r"\s*//.*", place_info):
                continue
            if re.match(r"\s*@\s*", place_info):
                self.curr_line_idx += 1
                break
            matcher = re.finditer(r"\S+", place_info)
            start = 0
            place_entity = PlaceEntity()
            place = None
            for m in matcher:
                place = m.group(0)
                place_entity.place_name = place
                start = m.end()
                break
            matcher = re.finditer(r"\d+", place_info)
            for m in matcher:
                if m.start() >= start:
                    start_marking[place] = m.group(0)
                    start = m.end()
                    break
            place_info = place_info[start:]
            pre_and_post = self._div_by_comma(place_info)
            pre = self._make_arcs(pre_and_post[0])
            post = self._make_arcs(pre_and_post[1])
            place_entity.pre = pre
            place_entity.post = post
            net_struct.append(place_entity)
        self.petri_net_file.net_struct = net_struct
        self.petri_net_file.map_info["startMarking"] = start_marking

    def _make_arcs(self, arcs_info):
        arcs = []
        if not arcs_info:
            return arcs
        arcs_info = self._delete_prefix_block(arcs_info)
        if not arcs_info:
            return arcs
        start = 0
        pattern_with_weight = re.compile(r"\S+\s*:\s*\S+")
        pattern_without_weight = re.compile(r"\S+")
        while start < len(arcs_info):
            arc_entity = ArcEntity()
            matcher_with_weight = pattern_with_weight.search(arcs_info, start)
            matcher_without_weight = pattern_without_weight.search(arcs_info, start)
            if not matcher_without_weight:
                break
            matcher_with_weight_start = matcher_with_weight.start() if matcher_with_weight else 2 ** 31 - 1
            matcher_without_weight_start = matcher_without_weight.start()
            if matcher_with_weight_start <= matcher_without_weight_start:
                start = matcher_with_weight.end()
                arc = matcher_with_weight.group(0)
                arc_info = arc.split(":")
                arc_info[0] = arc_info[0].replace(" ", "")
                arc_entity.tran_name = arc_info[0]
                arc_info[1] = arc_info[1].replace(" ", "")
                arc_entity.weight = int(arc_info[1])
            else:
                start = matcher_without_weight.end()
                arc = matcher_without_weight.group(0)
                arc_entity.tran_name = arc
                arc_entity.weight = 1
            arcs.append(arc_entity)
        return arcs

    def _make_map(self):
        s = self.lines[self.curr_line_idx]
        self.curr_line_idx += 1
        mp = {}
        s = self._get_data(s)
        pattern = re.compile(r"\S+\s*-\s*\S+")
        for match in pattern.finditer(s):
            entry = match.group(0)
            key_value = entry.split("-")
            key_value[0] = key_value[0].replace(" ", "")
            key_value[1] = key_value[1].replace(" ", "")
            if key_value[0] not in mp:
                mp[key_value[0]] = key_value[1]
            else:
                mp[key_value[0]] = mp[key_value[0]] + " " + key_value[1]
        return mp

    def _make_set(self):
        s = self.lines[self.curr_line_idx]
        self.curr_line_idx += 1
        st = set()
        s = self._get_data(s)
        for data in re.split(r"\s+", s):
            if data:
                st.add(data)
        return st

    def _make_value(self):
        s = self.lines[self.curr_line_idx]
        self.curr_line_idx += 1
        s = self._get_data(s)
        return int(s)

    def _is_map(self, s):
        return "-" in s

    def _is_value(self, s):
        return "=" in s

    def _is_ef(self, s):
        return s.strip().startswith("EF!")

    def _get_data(self, s):
        idx = s.index("=") + 1 if self._is_value(s) else s.index(":") + 1
        return s[idx:]

    def _div_by_comma(self, s):
        ss = s.split(",")
        ans0 = ss[0] if ss else ""
        ans1 = ss[1] if len(ss) == 2 else None
        return ans0, ans1

    def _delete_prefix_block(self, s):
        idx = 0
        while idx < len(s) and s[idx] == " ":
            idx += 1
        return s[idx:]


class SimpleXlsxReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.shared_strings = []
        self._load_shared_strings()

    def _load_shared_strings(self):
        try:
            with zipfile.ZipFile(self.file_path, "r") as zf:
                if "xl/sharedStrings.xml" not in zf.namelist():
                    return
                data = zf.read("xl/sharedStrings.xml")
            root = ET.fromstring(data)
            for si in root.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"):
                texts = [t.text or "" for t in si.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")]
                self.shared_strings.append("".join(texts))
        except Exception:
            self.shared_strings = []

    def _col_letters_to_index(self, letters):
        idx = 0
        for ch in letters:
            idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
        return idx

    def _cell_ref_to_col(self, cell_ref):
        letters = re.findall(r"[A-Z]+", cell_ref)
        return self._col_letters_to_index(letters[0]) if letters else 0

    def get_sheet_values(self, sheet_index):
        sheet_name = f"xl/worksheets/sheet{sheet_index}.xml"
        try:
            with zipfile.ZipFile(self.file_path, "r") as zf:
                if sheet_name not in zf.namelist():
                    return []
                data = zf.read(sheet_name)
        except Exception:
            return []
        root = ET.fromstring(data)
        rows = []
        ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        for row in root.findall(".//" + ns + "row"):
            cell_map = {}
            max_col = 0
            for cell in row.findall(ns + "c"):
                cell_ref = cell.attrib.get("r", "")
                col = self._cell_ref_to_col(cell_ref)
                max_col = max(max_col, col)
                cell_type = cell.attrib.get("t")
                value = ""
                if cell_type == "s":
                    v = cell.find(ns + "v")
                    if v is not None and v.text is not None:
                        idx = int(v.text)
                        if idx < len(self.shared_strings):
                            value = self.shared_strings[idx]
                elif cell_type == "inlineStr":
                    t = cell.find(".//" + ns + "t")
                    if t is not None and t.text is not None:
                        value = t.text
                else:
                    v = cell.find(ns + "v")
                    if v is not None and v.text is not None:
                        value = v.text
                cell_map[col] = value
            if max_col == 0:
                continue
            row_values = ["@"] * max_col
            for col, value in cell_map.items():
                if col > 0:
                    row_values[col - 1] = value if value != "" else "@"
            rows.append(row_values)
        return rows


class NeedMoveTranslator:
    def __init__(self, file_path):
        self.color_map = {}
        self.input_map = {}
        self.move_name_list = []
        self.color_id = {}
        self.move_id = {}
        try:
            reader = SimpleXlsxReader(file_path)
            a_list = self.get_list(reader.get_sheet_values(1))
            b_list = self.get_list(reader.get_sheet_values(2))
        except Exception:
            a_list = []
            b_list = []
        self.color_map = self.get_color_sort(a_list)
        j = 1
        for i in range(1, len(b_list)):
            l = self.get_move(b_list[i])
            if not l or not l[0]:
                continue
            self.input_map[b_list[i][0]] = l
            self.move_name_list.append(b_list[i][0])
            if b_list[i][0] == "COOLING":
                for k in range(1, 4):
                    self.move_id[b_list[i][0] + "3" + str(k)] = j
                    j += 1
                continue
            self.move_id[b_list[i][0]] = j
            j += 1

    def get_list(self, sheet_values):
        a_list = []
        for row in sheet_values:
            if not row or row[0] is None or row[0] == "":
                continue
            row_list = []
            for col in row:
                if col is None or col == "":
                    row_list.append("@")
                else:
                    row_list.append(str(col))
            a_list.append(row_list)
        return a_list

    def get_move(self, row_list):
        move_list = []
        for i in range(1, len(row_list)):
            moves = []
            s = self.get_string(row_list[i])
            k = 0
            while k < len(s):
                if s[k] == "(":
                    separate = []
                    value = s[k + 1]
                    k += 2
                    while k < len(s) and s[k] != ")":
                        separate.append(int(s[k]))
                        k += 1
                    move = Move(value, separate)
                    moves.append(move)
                else:
                    moves.append(Move(s[k]))
                k += 1
            move_list.append(moves)
        return move_list

    def get_color_sort(self, rows):
        count = 0
        mp = {}
        for string_list in rows:
            if string_list[0] == "颜色分类":
                continue
            lst = []
            m = string_list[0]
            for i in range(1, len(string_list)):
                s1 = "P" if i == 1 else "T"
                if string_list[i] == "@":
                    continue
                s = self.get_string(string_list[i])
                k = 0
                while k < len(s):
                    if s[k] == "(":
                        value = s[k + 1]
                        k += 2
                        while k < len(s) and s[k] != ")":
                            lst.append(s[k] + "(" + s1 + value + ")")
                            k += 1
                        k += 1
                    else:
                        lst.append(s[k] + "(" + s1 + ")")
                        k += 1
            mp[tuple(lst)] = m
            self.color_id[m] = count
            count += 1
        return mp

    def get_string(self, s):
        lst = []
        i = 0
        while i < len(s):
            if s[i] == "(":
                lst.append(s[i])
                i += 1
                continue
            if s[i].isdigit():
                str_buf = []
                while i < len(s) and s[i].isdigit():
                    str_buf.append(s[i])
                    i += 1
                if str_buf:
                    lst.append("".join(str_buf))
                continue
            if s[i] == ")":
                lst.append(s[i])
            i += 1
        return lst
