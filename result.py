import os
import sys

class Result(object):
    def __init__(self):
        self.entity = []
        self.date = []
        self.number = []
        self.metric = []

    def get_result(self, tokens, tags):
        return self.result_to_json(tokens, tags)

    def result_to_json(self, tokens, tags):
        mp = {"ENTITY": [], "DATE": [], "NUMBER": [], "METRIC": []}
        prev_name = ""
        prev_value = []
        for token, tag in zip(tokens, tags):
            arr = tag.strip().split("-")  
            if arr[0] == "B":
                if prev_name != "" and len(prev_value) > 0:
                    mp[prev_name].append("".join(prev_value))
                prev_name = arr[1]
                prev_value = [token]
            elif arr[0] == "I":
                if prev_name != arr[1]:
                    if prev_name != "" and len(prev_value) > 0:
                        mp[prev_name].append("".join(prev_value))
                    #filter invalid tag
                    prev_name, prev_value = "", []
                    continue
                prev_value.append(token)
            elif arr[0] == "O":
                if prev_name != "" and len(prev_value) > 0:
                    mp[prev_name].append("".join(prev_value))
                prev_name, prev_value = "", []
            elif arr[0] == "X":
                # 跟随前一个label
                if prev_name != "" and len(prev_value) > 0:
                    prev_value.append(token) 
                else:
                    prev_name, prev_value = "", []
        if prev_name != "" and len(prev_value) > 0:
                mp[prev_name].append("".join(prev_value))

        return mp
