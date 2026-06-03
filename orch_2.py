import anthropic

import openai

import fireworks

import re
import json
import json_repair


import os
from collections import deque

import numpy as np


import time
from pydantic import BaseModel
from pathlib import Path
from pydantic import TypeAdapter
import typing as T


from copy import deepcopy

import logging
logging.basicConfig(filename="run_log.txt", level=logging.INFO)


from render import *

import argparse
import threading
import concurrent.futures as _cf

from pricing import (
    BudgetTracker,
    BudgetExceeded,
    resolve_budget,
    blended_price_per_mtok,
    price_for,
    load_price_file,
)


def _usage_tokens(resp) -> tuple[int, int]:
    """Best-effort (input_tokens, output_tokens) from any provider response.

    Handles Anthropic Messages (usage.input_tokens/output_tokens),
    OpenAI Responses (usage.input_tokens/output_tokens) and OpenAI-compatible
    Chat Completions (usage.prompt_tokens/completion_tokens).
    """
    u = getattr(resp, "usage", None)
    if u is None:
        return 0, 0
    in_tok = getattr(u, "input_tokens", None)
    if in_tok is None:
        in_tok = getattr(u, "prompt_tokens", 0)
    out_tok = getattr(u, "output_tokens", None)
    if out_tok is None:
        out_tok = getattr(u, "completion_tokens", 0)
    return int(in_tok or 0), int(out_tok or 0)


GRID = list[list[int]]



class Example(BaseModel):
    input: GRID
    output: GRID


class Challenge(BaseModel):
    id: str
    train: list[Example]
    test: list[Example]






#hyperparameter responsible for switching between the largest cluster and the cluster that has the closest unsolved 
# task closer to its center
p = 0.7

#hyperparameter cluster size
#n = 10

n = 10


ChallengeAdapter = TypeAdapter(dict[str, Challenge])




class ColorData(BaseModel):
    name: str
    hex: str
    rgb: tuple[int, int, int]


color_map: dict[int, ColorData] = {
    0: ColorData(name="black", hex="#000000", rgb=(0, 0, 0)),
    1: ColorData(name="blue", hex="#0074D9", rgb=(0, 40, 230)),
    2: ColorData(name="red", hex="#FF4136", rgb=(230, 20, 20)),
    3: ColorData(name="green", hex="#2ECC40", rgb=(46, 204, 64)),
    4: ColorData(name="yellow", hex="#FFDC00", rgb=(255, 255, 0)),
    5: ColorData(name="grey", hex="#AAAAAA", rgb=(170, 170, 170)),
    6: ColorData(name="pink", hex="#F012BE", rgb=(255, 0, 195)),
    7: ColorData(name="orange", hex="#FF851B", rgb=(255, 133, 27)),
    8: ColorData(
        name="purple", hex="#9d00ff", rgb=(157, 0, 255)
    ),  # is sky blue on normal visuals
    9: ColorData(name="brown", hex="#870C25", rgb=(139, 69, 19)),
}


spreadsheet_col_labels = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "AA",
    "AB",
    "AC",
    "AD",
]

def spreadsheet_ascii_grid(grid: np.ndarray, separator: str = "|") -> str:
    rows, cols = grid.shape
    assert cols <= 30
    assert rows <= 30

    cols_header_line = separator.join([" "] + spreadsheet_col_labels[:cols])
    rest = "\n".join(
        separator.join([str(i + 1)] + [str(x) for x in row])
        for i, row in enumerate(grid)
    )

    return f"{cols_header_line}\n{rest}"


def grid_to_ascii(
    grid: np.ndarray, separator: str = "|", spreadsheet_ascii: bool = False
) -> str:
    if spreadsheet_ascii:
        return spreadsheet_ascii_grid(grid, separator=separator)

    return "\n".join(separator.join(str(x) for x in row) for row in grid)


def array_to_str(grid: GRID) -> str:
    final_output = "["
    for i, row in enumerate(grid):
        final_output += f"{str(row)}"
        if i != len(grid) - 1:
            final_output += ","
    final_output += "]"
    return final_output


def grid_diffs_to_ascii(
    grid_input: np.ndarray, grid_output: np.ndarray, separator: str = "|"
) -> str:
    assert grid_input.shape == grid_output.shape
    row_nums = grid_input.shape[0]
    col_nums = grid_input.shape[1]
    diff_arr: list[list[str]] = [
        ["  --  " for col in range(col_nums)] for row in range(row_nums)
    ]
    # iterate through the rows and columns using the shape
    for row_ind in range(row_nums):
        for col_ind in range(col_nums):
            if grid_input[row_ind][col_ind] != grid_output[row_ind][col_ind]:
                diff_arr[row_ind][col_ind] = (
                    f"{grid_input[row_ind][col_ind]} -> {grid_output[row_ind][col_ind]}"
                )

    return "\n".join(separator.join(str(x) for x in row) for row in diff_arr)


def get_spreadsheet_notation_str(i, j, quote: bool = True):
    out = f"{spreadsheet_col_labels[j]}{i+1}"
    if quote:
        out = f'"{out}"'
    return out


def get_spreadsheet_notation_support_runs(rows_cols: list[tuple[int, int]]):
    row_cols_v = np.array(sorted(rows_cols, key=lambda x: (x[0], x[1])))

    running_str = ""

    idx = 0
    while idx < len(row_cols_v):
        r, c = row_cols_v[idx]

        count_in_a_row = 0
        for checking_idx, (n_r, n_c) in enumerate(row_cols_v[idx:]):
            if n_r == r and n_c == c + checking_idx:
                count_in_a_row += 1
            else:
                break

        if count_in_a_row > 4:
            start = get_spreadsheet_notation_str(r, c, quote=False)
            c_end = c + count_in_a_row - 1

            assert np.array_equal(row_cols_v[idx + count_in_a_row - 1], (r, c_end)), (
                row_cols_v[idx + count_in_a_row - 1],
                (r, c_end),
            )

            end = get_spreadsheet_notation_str(r, c_end, quote=False)

            running_str += f" {start} ... {end}"
            idx += count_in_a_row
        else:
            running_str += " " + get_spreadsheet_notation_str(r, c, quote=False)
            idx += 1

    return running_str



def content_blocks_from_matrix(
    *,
    matrix: GRID,
    _label: str,
    include_image: bool,
    use_ascii: bool,
    use_array: bool,
) -> list[dict[str, str]]:
    matrix = deepcopy(matrix)
    grid = np.array(matrix)
    x, y = grid.shape
    messages = [
        {"type": "text", "text": _label},
        {"type": "text", "text": f"Shape: {x} by {y}\n\n"},
    ]
    if include_image:
        messages.append(grid_to_base64_png_oai_content(grid=grid))
    if use_ascii:
        messages.append(
            {
                "type": "text",
                "text": f"ASCII representation:\n\n{grid_to_ascii(grid=grid, separator='|', spreadsheet_ascii=False)}\n\n",
            }
        )
    if use_array:
        messages.append({"type": "text", "text":f"array representation:\n\n{array_to_str(grid=matrix)}\n\n"})
    return messages


def does_grid_change_shape(challenge: Challenge) -> bool:
    for train in challenge.train:
        if np.array(train.input).shape != np.array(train.output).shape:
            return True
    return False



def content_from_challenge(
    challenge: Challenge,
    include_diffs: bool,
    include_image: bool,
    use_ascii: bool,
    use_array: bool,
) -> list[dict[str, T.Any]]:
    content = []
    for i, train in enumerate(challenge.train):
        example_number = i + 1
        # add input blocks
        content.extend(
            content_blocks_from_matrix(
                matrix=train.input,
                _label=f"# Example {example_number}\n\n## Input {example_number}\n\n",
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
        )
        # add output blocks
        content.extend(
            content_blocks_from_matrix(
                matrix=train.output,
                _label=f"## Output {example_number}\n\n",
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
        )
        if not does_grid_change_shape(challenge=challenge) and include_diffs:
            content.append(
                {
                    "type": "text",
                    "text": f"## Color changes between the Input and Output ASCII representation:\n\n"
                    f"{grid_diffs_to_ascii(grid_input=np.array(train.input), grid_output=np.array(train.output), separator='|')}\n\n",
                }
            )

    # TODO for now, only do the first test... Will have to treat these multi tests as multiple examples later
    # assert len(challenge.test) == 1
    content.extend(
        content_blocks_from_matrix(
            matrix=challenge.test[0].input,
            _label="# Additional input\n\n",
            include_image=include_image,
            use_ascii=use_ascii,
            use_array=use_array,
        )
    )

    return content

# NEXT : write the main orchestrator loop
# NEXT NEXT

# NEXT self.get_tasks, self.run_code

class Orchestrator:
    def __init__(self, writer: str, coder: str, selector: str, api_key_writer: str, api_key_coder: str, api_key_selector: str, num_tasks: int = 120, taskdir: str = "/ARC-AGI-2/data/training/", num_retries: int = 4, max_workers: int = 8, max_output_tokens: int | None = None, budget: "BudgetTracker | None" = None):

        # Each role -> (client, model_id, provider_str).  Provider is tracked
        # explicitly (instead of isinstance checks) so that Fireworks served
        # over an OpenAI-compatible endpoint is distinguishable, and so the
        # cloud subclass can plug in its own backend.
        self._roles: dict[str, tuple] = {}
        self._roles["writer"] = self._make_role(writer, api_key_writer)
        self._roles["coder"] = self._make_role(coder, api_key_coder)

        # Backwards-compatible attributes used throughout the class.
        self.writer, self.writermodel, self._writer_provider = self._roles["writer"]
        self.coder, self.codermodel, self._coder_provider = self._roles["coder"]

        if selector == "hemens":
            self.selector = "hemens"
            self._roles["selector"] = ("hemens", "hemens", "hemens")
        else:
            self._roles["selector"] = self._make_role(selector, api_key_selector)
            self.selector = self._roles["selector"][0]
            self.selectormodel = selector

        self.num_tasks = num_tasks
        self.solved = [False for _ in range(self.num_tasks)]
        self.taskdir = str(os.getcwd()) + taskdir
        self.num_retries = num_retries

        # --- new: concurrency + budget ---------------------------------
        self.max_workers = max(1, int(max_workers))
        self.max_output_tokens = max_output_tokens  # caps every generation
        self.budget = budget                        # BudgetTracker or None
        self._lock = threading.Lock()               # guards shared state
        self._succ_compress_cache: dict[int, str] = {}  # avoid recompressing

    # ------------------------------------------------------------------
    # Provider plumbing (overridable by CloudOrchestrator)
    # ------------------------------------------------------------------
    def _make_role(self, model: str, api_key: str) -> tuple:
        """Return (client, model, provider) routed by model-name prefix."""
        if model[:3] == "gpt" or model[:2] in ("o1", "o3", "o4"):
            return (openai.OpenAI(api_key=api_key), model, "openai")
        elif model[:4] in ["haik", "sonn", "opus", "anth", "clau"]:
            return (anthropic.Anthropic(api_key=api_key), model, "anthropic")
        else:
            # Fireworks via its OpenAI-compatible endpoint.  The bare
            # `fireworks` pip package does not always expose a usable client
            # class, whereas the OpenAI SDK pointed at Fireworks always works.
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.fireworks.ai/inference/v1",
            )
            return (client, model, "fireworks")

    def _raw_call(self, client, model, provider, content, max_tokens):
        """Make ONE provider call. Returns (text, input_tokens, output_tokens).

        Centralised so token usage can be metered for the budget and so the
        cloud subclass only needs to override this one method.
        """
        if provider == "anthropic":
            resp = client.messages.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
            )
            text = resp.content[0].text
        elif provider == "openai":
            resp = client.responses.create(
                model=model,
                input=content,
                max_output_tokens=max_tokens,
            )
            text = resp.output_text
        else:  # fireworks / generic OpenAI-compatible chat endpoint
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content
        in_tok, out_tok = _usage_tokens(resp)
        return text, in_tok, out_tok

    def _complete(self, which: str, content, max_tokens: int) -> str:
        """Budget-aware single completion for role `which` (writer/coder/selector)."""
        client, model, provider = self._roles[which]
        if self.budget is not None and self.budget.would_exceed():
            raise BudgetExceeded(
                f"USD budget reached before {which} call: {self.budget.summary()}"
            )
        cap = max_tokens
        if self.max_output_tokens:
            cap = min(cap, self.max_output_tokens)
        text, in_tok, out_tok = self._raw_call(client, model, provider, content, cap)
        if self.budget is not None:
            self.budget.record(model, in_tok, out_tok)
        return text

    @staticmethod
    def _as_text(x) -> str:
        """Coerce anything (str / pydantic model / list of blocks) to text.

        Used when *appending context* to a prompt — the original code did
        `prompt += challenge` / `prompt += list_of_blocks`, which raises
        TypeError (see README_UPGRADE.md, 'compression & context').
        """
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, BaseModel):
            return x.model_dump_json()
        try:
            return json.dumps(x, default=str)
        except Exception:
            return str(x)

    def _parallel_map(self, fn, items):
        """Run fn(item) over items in threads, preserving input order.

        Returns a list aligned with `items` (None where a worker raised or was
        skipped). Stops submitting new work once the budget is exhausted.
        """
        items = list(items)
        results = [None] * len(items)
        if self.max_workers <= 1:
            for i, it in enumerate(items):
                if self.budget is not None and self.budget.would_exceed():
                    break
                try:
                    results[i] = fn(it)
                except BudgetExceeded:
                    break
                except Exception as e:  # keep going on per-item failure
                    print("worker error:", e)
            return results
        with _cf.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            fut_to_i = {}
            for i, it in enumerate(items):
                if self.budget is not None and self.budget.would_exceed():
                    break
                fut_to_i[ex.submit(fn, it)] = i
            for fut in _cf.as_completed(fut_to_i):
                i = fut_to_i[fut]
                try:
                    results[i] = fut.result()
                except BudgetExceeded:
                    pass
                except Exception as e:
                    print("worker error:", e)
        return results

    def _cluster_or_random(self, taskprompts, formatted):
        """select_cluster wrapped with a random-cluster fallback.

        The clustering math in select_cluster raises (e.g. zero-size reduction)
        when the selected set is all-solved or all-unsolved; rather than kill
        the whole run we fall back to a random cluster.
        """
        try:
            idxs = np.asarray(self.select_cluster(taskprompts, formatted)).ravel().astype(int)
            if idxs.size == 0:
                raise ValueError("empty cluster")
            return idxs
        except Exception as e:
            print("select_cluster failed, using random cluster:", e)
            return np.asarray(
                np.random.randint(0, self.num_tasks, size=n)
            ).ravel().astype(int)

    def build_challenges_v2(
        self,
        challenges_path: Path
    ) -> dict[str, Challenge]:
        # Read every file in the directory and use the file_path prefix as the key in the challenges_j dictionary
        challenges_j = {}
        for file_path in sorted(challenges_path.iterdir()):
            if file_path.is_file() and file_path.suffix == ".json":
                with open(file_path) as f:
                    file_challenge = json.load(f)
                    # Use the file name without suffix as the key
                    key = file_path.stem
                    challenges_j[key] = file_challenge
                    challenges_j[key]["id"] = key
        return ChallengeAdapter.validate_python(challenges_j)

    def extract_rule(self, desc):
        match = re.search(r'={10}\s*\nTRANSFORMATION RULE:\s*\n(.*?)={10}', desc, re.DOTALL)
        return match.group(1).strip() if match else desc

    def compress_criticism(self, desc: str):
        prompt = f""" below are the examples of previous unsuccessful runs, the programs that were \
        executed during these unsuccessful runs, and the criticism of what was done wrong. Summarize \
        each unsuccessful run and the criticism without losing significant details, but try to keep \
        the information pertaining to each unsuccessful run below 150, maximum 250 tokens"""
        prompt += desc
        return self.call_writer(prompt) 
    

    def compress_successful(self, desc: str, code: str):
        prompt = f"""below are the examples of tasks that were successfully solved, their solutions and, if \
        the solution was successfully solved not from the first attempt -- history of criticism of unsuccessful attempts \

        Summarize each successful run and the criticism without losing significant details, but try to keep \
        the information pertaining to each successful run below 200, maximum 400 tokens"""
        prompt += "\nDESCRIPTION: " + desc
        prompt += "\nCODE: " + code
        return self.call_writer(prompt) 




      

    def call_writer(self, prompt, max_tokens: int = 8192):
        # number of attempts:
        content = ""
        for _ in range(3):
            try:
                content = self._complete("writer", prompt, max_tokens)
                if content:
                    return content
            except BudgetExceeded:
                raise
            except Exception as e:
                print("Error in call_writer:", e)
        return content
        
    
    
    def write_self_critique(self, task, prompt, code, output, json_task):
        critiqueprompt = f"""
        You are given a task description, an unsuccessful attempt, and maybe the chronology of previous unsuccessful attempts critiques. You need to write the critique of the last unsuccesful attempt.
        This is the task description: {task}
        This is the original task in json format: {json_task}
        This is the description of the task, and maybe the chronology of critique of previous attempts to solve this task: {prompt}.
        This is the last unsuccessful attempt{code}, and the resulting output{output}
        Looking at the task, its description, maybe the chronology of critique of the previous attempts, and especially at 
        the last unsuccessful attempt and the resulting output of this attempt, highlight what does the person or agent who solves it does wrong, and what can be improved,
        what can the person do better in solving this task?
        Limit the critique to 300, if you can do so without losing important information, maximum 600 words.
        """
        critique = self.call_writer(critiqueprompt, max_tokens = 2048)
        print("=====================================================")
        print("CRITIQUELEN ", len(critique))
        print("CRITIQUE ", critique)
        print("=====================================================")
        return critique


  
    
    #def write_description(self, tasks):
    #    descs = []
    #    for task in tasks:
    #        print("======================================")
    #        print("WHAT WE ARE WRITTING PROMPT TO ", task)
    #        print("======================================")
    #        prompt = """You will be given some number of paired example inputs and outputs. The outputs were produced by applying a transformation rule to the inputs. Your task is to determine the transformation rule and describe it proactively and thoroughly.\n\nThe inputs and outputs are each \"grids\". A grid is a rectangular matrix of integers between 0 and 9 (inclusive). These grids will be shown to you as grids of numbers (ASCII). The the grid of numbers corresponds to the coloured image is as follows: black: 0, blue: 1, red: 2, green: 3, yellow: 4, grey: 5, pink: 6, orange: 7, purple: 8, brown: 9.\n\nThe transformation rule maps from each input to a single correct output, so that a potential implementation in code must be exactly correct. Thus, you need to resolve all potential uncertainties you might have about the transformation rule. For instance, if the examples always involve some particular color being changed to another color in the output, but which color it is changed to varies between different examples, then you need to figure out what determines the correct output color. As another example, if some shape(s) or cells in the input are relocated or recolored, you need to determine which exact shapes should be relocated/recolored in the output and where they should be moved or what their color in the output should be. Whenever there are potential ambiguities/uncertainties in your current understanding of the transformation rule, you need to resolve them. You should resolve ambiguities and uncertainties by carefully analyzing the examples and using step by step reasoning.\n\nThe transformation rule might have multiple components and might be fairly complex. It's also reasonably common that the transformation rule has one main rule (e.g., replace cells in XYZ pattern with color ABC), but has some sort of exception (e.g., don't replace cells if they have color DEF). So, you should be on the lookout for additional parts or exceptions that you might have missed so far. Consider explicitly asking yourself (in writing): \"Are there any additional parts or exceptions to the transformation rule that I might have missed?\" (Rules don't necessarily have multiple components or exceptions, but it's common enough that you should consider it.)\n\nHere are some examples of transformation rules with multiple components or exceptions:\n\n- There is a grey grid with black holes that have different shapes and the rule is to fill in these holes with colored cells. Further, the color to use for each hole depends on the size of the hole (in terms of the number of connected cells). 1 cell holes are filled with pink, 2 cell holes are filled with blue, and 3 cell holes are filled with red.\n- The output is 3x3 while the input is 3x7. The output has red cells while the input has two \"sub-grids\" that are 3x3 and separated by a grey line in the middle. Each of the sub-grids has some colored cells (blue) and some black cells. The rule is to AND the two sub-grids together (i.e., take the intersection of where the two sub-grids are blue) and color the 3x3 cells in the output red if they are in the intersection and black otherwise.\n- The grey rectangular outlines are filled with some color in the output. Pink, orange, and purple are used to fill in the voids in different cases. The color depends on the size of the black void inside the grey outline where it is pink if the void has 1 cell (1x1 void), orange if the gap has 4 cells, and purple if the gap was 9 cells. For each void, all of the filled-in colors are the same.\n- The red shape in the input is moved. It is moved either horizontally or vertically. It is moved until moving it further would intersect with a purple shape.
    #        It is moved in the direction of the purple shape, that is, moved in whichever direction would involve it eventually intersecting with this purple shape.\n\nThese are just example rules; the actual transformation rule will be quite different. But, this should hopefully give you some sense of what transformation rules might look like.\n\nNote that in each of these cases, you would need to find the rule by carefully examining the examples and using reasoning. You would need to describe the transformation rule precisely and exhaustively, taking into account all possible cases and getting all of the details right (e.g., exactly where to place various things or exactly which color to use in each case). If the details aren't fully ironed out, you should do additional reasoning to do so before giving the answer .\n\nYou'll need to carefully reason in order to determine the transformation rule. Start your response by carefully reasoning in <reasoning></reasoning> tags.\n You follow a particular reasoning style. You break down complex problems into smaller parts and reason through them step by step, arriving at sub-conclusions before stating an overall conclusion. This reduces the extent to which you need to do large leaps of reasoning.\n\nYou reason in substantial detail for as long as is necessary to fully determine the transformation rule and resolve any ambiguities/uncertainties.
    #        The task itself is:{task}
    #        """
    #        desc = self.call_writer(prompt)
    #        descs.append(desc)
    #    return descs
    
    #def write_description(self, tasks, formatted_blocks):
    #    descs = []
    #    for task, blocks in zip(tasks, formatted_blocks):
    #        prompt = f"""You will be given some number of paired example inputs and outputs. The outputs were produced by applying a transformation rule to the inputs. Your task is to determine the transformation rule and describe it proactively and thoroughly.\n\nThe inputs and outputs are each \"grids\". A grid is a rectangular matrix of integers between 0 and 9 (inclusive). These grids will be shown to you as grids of numbers (ASCII). The the grid of numbers corresponds to the coloured image is as follows: black: 0, blue: 1, red: 2, green: 3, yellow: 4, grey: 5, pink: 6, orange: 7, purple: 8, brown: 9.\n\nThe transformation rule maps from each input to a single correct output, so that a potential implementation in code must be exactly correct. Thus, you need to resolve all potential uncertainties you might have about the transformation rule. For instance, if the examples always involve some particular color being changed to another color in the output, but which color it is changed to varies between different examples, then you need to figure out what determines the correct output color. As another example, if some shape(s) or cells in the input are relocated or recolored, you need to determine which exact shapes should be relocated/recolored in the output and where they should be moved or what their color in the output should be. Whenever there are potential ambiguities/uncertainties in your current understanding of the transformation rule, you need to resolve them. You should resolve ambiguities and uncertainties by carefully analyzing the examples and using step by step reasoning.\n\nThe transformation rule might have multiple components and might be fairly complex. It's also reasonably common that the transformation rule has one main rule (e.g., replace cells in XYZ pattern with color ABC), but has some sort of exception (e.g., don't replace cells if they have color DEF). So, you should be on the lookout for additional parts or exceptions that you might have missed so far. Consider explicitly asking yourself (in writing): \"Are there any additional parts or exceptions to the transformation rule that I might have missed?\" (Rules don't necessarily have multiple components or exceptions, but it's common enough that you should consider it.)\n\nHere are some examples of transformation rules with multiple components or exceptions:\n\n- There is a grey grid with black holes that have different shapes and the rule is to fill in these holes with colored cells. Further, the color to use for each hole depends on the size of the hole (in terms of the number of connected cells). 1 cell holes are filled with pink, 2 cell holes are filled with blue, and 3 cell holes are filled with red.\n- The output is 3x3 while the input is 3x7. The output has red cells while the input has two \"sub-grids\" that are 3x3 and separated by a grey line in the middle. Each of the sub-grids has some colored cells (blue) and some black cells. The rule is to AND the two sub-grids together (i.e., take the intersection of where the two sub-grids are blue) and color the 3x3 cells in the output red if they are in the intersection and black otherwise.\n- The grey rectangular outlines are filled with some color in the output. Pink, orange, and purple are used to fill in the voids in different cases. The color depends on the size of the black void inside the grey outline where it is pink if the void has 1 cell (1x1 void), orange if the gap has 4 cells, and purple if the gap was 9 cells. For each void, all of the filled-in colors are the same.\n- The red shape in the input is moved. It is moved either horizontally or vertically. It is moved until moving it further would intersect with a purple shape.
    #        It is moved in the direction of the purple shape, that is, moved in whichever direction would involve it eventually intersecting with this purple shape.\n\nThese are just example rules; the actual transformation rule will be quite different. But, this should hopefully give you some sense of what transformation rules might look like.\n\nNote that in each of these cases, you would need to find the rule by carefully examining the examples and using reasoning. You would need to describe the transformation rule precisely and exhaustively, taking into account all possible cases and getting all of the details right (e.g., exactly where to place various things or exactly which color to use in each case). If the details aren't fully ironed out, you should do additional reasoning to do so before giving the answer .\n\nYou'll need to carefully reason in order to determine the transformation rule. Start your response by carefully reasoning in <reasoning></reasoning> tags.\n You follow a particular reasoning style. You break down complex problems into smaller parts and reason through them step by step, arriving at sub-conclusions before stating an overall conclusion. This reduces the extent to which you need to do large leaps of reasoning.\n\nYou reason in substantial detail for as long as is necessary to fully determine the transformation rule and resolve any ambiguities/uncertainties.
    #        The task itself is:{task}
    #        """
    #        
    #        # Bilder aus den Blöcken extrahieren
    #        images = [b for b in blocks if b["type"] == "image_url"]
    #        
    #        if images and isinstance(self.writer, anthropic.Anthropic):
    #            # Anthropic Format: image_url → image mit base64 source
    #            content = []
    #            for img in images:
    #                url = img["image_url"]["url"]
    #                base64_data = url.split(",")[1]
    #                content.append({
    #                    "type": "image",
    #                    "source": {
    #                       "type": "base64",
    #                        "media_type": "image/png",
    #                        "data": base64_data,
    #                    }
    #                })
    #            content.append({"type": "text", "text": prompt})
    #            desc = self.call_writer(content)
    #            print("UNSER BESCHRIEBUNG ", desc)
    #       else:
    #            desc = self.call_writer(prompt)
    #        
    #        descs.append(desc)
    #    return descs

    def re_describe(self, task, formatted_block, jsons, critique, succ_desc, succ_code ):
        prompt = f"""You will be given some number of paired example inputs and outputs.
        The outputs were produced by applying a transformation rule to the inputs.
        Your task is to determine the transformation rule and describe it proactively and thoroughly.
        \n\n The inputs and outputs are each \"grids\". A grid is a rectangular matrix of integers between 0 and 9 (inclusive).
        These grids will be shown to you as grids of numbers (ASCII). 
        The the grid of numbers corresponds to the coloured image is as follows: black: 0, blue: 1, red: 2, green: 3, yellow: 4, grey: 5, pink: 6, orange: 7, purple: 8, brown: 9.\n\n
        The transformation rule maps from each input to a single correct output, so that a potential implementation in code must be exactly correct. 
        Thus, you need to resolve all potential uncertainties you might have about the transformation rule. 
        For instance, if the examples always involve some particular color being changed to another color in the output, but which color it is changed to varies between different examples, then you need to figure out what determines the correct output color. 
        As another example, if some shape(s) or cells in the input are relocated or recolored, you need to determine which exact shapes should be relocated/recolored in the output and where they should be moved or what their color in the output should be. 
        Whenever there are potential ambiguities/uncertainties in your current understanding of the transformation rule, you need to resolve them. 
        You should resolve ambiguities and uncertainties by carefully analyzing the examples and using step by step reasoning.\n\n
        The transformation rule might have multiple components and might be fairly complex. It's also reasonably common that the transformation rule has one main rule (e.g., replace cells in XYZ pattern with color ABC), but has some sort of exception (e.g., don't replace cells if they have color DEF). So, you should be on the lookout for additional parts or exceptions that you might have missed so far. Consider explicitly asking yourself (in writing): \"Are there any additional parts or exceptions to the transformation rule that I might have missed?\" (Rules don't necessarily have multiple components or exceptions, but it's common enough that you should consider it.)\n\nHere are some examples of transformation rules with multiple components or exceptions:\n\n- There is a grey grid with black holes that have different shapes and the rule is to fill in these holes with colored cells. Further, the color to use for each hole depends on the size of the hole (in terms of the number of connected cells). 1 cell holes are filled with pink, 2 cell holes are filled with blue, and 3 cell holes are filled with red.\n- The output is 3x3 while the input is 3x7. The output has red cells while the input has two \"sub-grids\" that are 3x3 and separated by a grey line in the middle. Each of the sub-grids has some colored cells (blue) and some black cells. The rule is to AND the two sub-grids together (i.e., take the intersection of where the two sub-grids are blue) and color the 3x3 cells in the output red if they are in the intersection and black otherwise.\n- The grey rectangular outlines are filled with some color in the output. Pink, orange, and purple are used to fill in the voids in different cases. The color depends on the size of the black void inside the grey outline where it is pink if the void has 1 cell (1x1 void), orange if the gap has 4 cells, and purple if the gap was 9 cells. For each void, all of the filled-in colors are the same.\n- The red shape in the input is moved. It is moved either horizontally or vertically. It is moved until moving it further would intersect with a purple shape. It is moved in the direction of the purple shape, that is, moved in whichever direction would involve it eventually intersecting with this purple shape.\n\nThese are just example rules; the actual transformation rule will be quite different. But, this should give you some sense of what transformation rules might look like.\n\n
        Note that in each of these cases, you would need to find the rule by carefully examining the examples and using reasoning. 
        You would need to describe the transformation rule precisely and exhaustively, taking into account all possible cases and getting all of the details right (e.g., exactly where to place various things or exactly which color to use in each case). If the details aren't fully ironed out, you should do additional reasoning to do so before giving the answer .\n\n
        You'll need to reason in chains of thought in order to determine the transformation rule. Start your response by carefully reasoning in <reasoning></reasoning> tags.\n You follow a particular reasoning style. You break down complex problems into smaller parts and reason through them step by step, arriving at sub-conclusions before stating an overall conclusion. This reduces the extent to which you need to do large leaps of reasoning.\n\nYou reason in substantial detail for as long as is necessary to fully determine the transformation rule and resolve any ambiguities/uncertainties.
        The task itself in the ASCII, diff between input and output and array format is:{task}
        The original task in json format is {jsons}
        After you finished all your deliberations, skip two lines,write the sign \"=\" ten times, from one more new line write \"TRANSFORMATION RULE: \" and, from one more new line, state the transformation rule to which you arrived as a result of your deliberations again, followed by ten more \"=\" signs.
        Previously, you have unsuccessfully described/solved this task. The critique of your previous attempt is {critique}
        Also, previously, you have successfully described a task that has some similarities with this task, and have successfully written code for it. The description of this task was as follows: 
        =====================SUCCESSFUL TASK DESCRIPTIONS FOR TASKS THAT HAVE SIMILARITIES WITH THE CURRENT ONE====================================
        {succ_desc}, and
        the successful code for that task that you solved correctly was as follows: 
        =======================SUCCESSFUL CODE FOR TASKS THAT HAVE SIMILARITIES WITH THE CURRENT ONE====================================
        {succ_code}
        """
        
        # Build interleaved content: prompt first, then all blocks in order
        content = [{"type": "text", "text": prompt}]
        
        for block in formatted_block:
            if block["type"] == "image_url" and isinstance(self.writer, anthropic.Anthropic):
                url = block["image_url"]["url"]
                base64_data = url.split(",")[1]
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_data,
                    }
                })
            #else:
            #    content.append(block)
        
        desc = self.call_writer(content)
        logging.info(f"task description : {desc}")
        return desc

    def write_description(self, tasks, formatted_blocks, jsons):
        items = list(zip(tasks, formatted_blocks, jsons))

        def _one(it):
            task, blocks, json_task = it
            prompt = f"""You will be given some number of paired example inputs and outputs.
            The outputs were produced by applying a transformation rule to the inputs.
            Your task is to determine the transformation rule and describe it proactively and thoroughly.
            \n\n The inputs and outputs are each \"grids\". A grid is a rectangular matrix of integers between 0 and 9 (inclusive).
            These grids will be shown to you as grids of numbers (ASCII). 
            The the grid of numbers corresponds to the coloured image is as follows: black: 0, blue: 1, red: 2, green: 3, yellow: 4, grey: 5, pink: 6, orange: 7, purple: 8, brown: 9.\n\n
            The transformation rule maps from each input to a single correct output, so that a potential implementation in code must be exactly correct. 
            Thus, you need to resolve all potential uncertainties you might have about the transformation rule. 
            For instance, if the examples always involve some particular color being changed to another color in the output, but which color it is changed to varies between different examples, then you need to figure out what determines the correct output color. 
            As another example, if some shape(s) or cells in the input are relocated or recolored, you need to determine which exact shapes should be relocated/recolored in the output and where they should be moved or what their color in the output should be. 
            Whenever there are potential ambiguities/uncertainties in your current understanding of the transformation rule, you need to resolve them. 
            You should resolve ambiguities and uncertainties by carefully analyzing the examples and using step by step reasoning.\n\n
            The transformation rule might have multiple components and might be fairly complex. It's also reasonably common that the transformation rule has one main rule (e.g., replace cells in XYZ pattern with color ABC), but has some sort of exception (e.g., don't replace cells if they have color DEF). So, you should be on the lookout for additional parts or exceptions that you might have missed so far. Consider explicitly asking yourself (in writing): \"Are there any additional parts or exceptions to the transformation rule that I might have missed?\" (Rules don't necessarily have multiple components or exceptions, but it's common enough that you should consider it.)\n\nHere are some examples of transformation rules with multiple components or exceptions:\n\n- There is a grey grid with black holes that have different shapes and the rule is to fill in these holes with colored cells. Further, the color to use for each hole depends on the size of the hole (in terms of the number of connected cells). 1 cell holes are filled with pink, 2 cell holes are filled with blue, and 3 cell holes are filled with red.\n- The output is 3x3 while the input is 3x7. The output has red cells while the input has two \"sub-grids\" that are 3x3 and separated by a grey line in the middle. Each of the sub-grids has some colored cells (blue) and some black cells. The rule is to AND the two sub-grids together (i.e., take the intersection of where the two sub-grids are blue) and color the 3x3 cells in the output red if they are in the intersection and black otherwise.\n- The grey rectangular outlines are filled with some color in the output. Pink, orange, and purple are used to fill in the voids in different cases. The color depends on the size of the black void inside the grey outline where it is pink if the void has 1 cell (1x1 void), orange if the gap has 4 cells, and purple if the gap was 9 cells. For each void, all of the filled-in colors are the same.\n- The red shape in the input is moved. It is moved either horizontally or vertically. It is moved until moving it further would intersect with a purple shape. It is moved in the direction of the purple shape, that is, moved in whichever direction would involve it eventually intersecting with this purple shape.\n\nThese are just example rules; the actual transformation rule will be quite different. But, this should give you some sense of what transformation rules might look like.\n\n
            Note that in each of these cases, you would need to find the rule by carefully examining the examples and using reasoning. 
            You would need to describe the transformation rule precisely and exhaustively, taking into account all possible cases and getting all of the details right (e.g., exactly where to place various things or exactly which color to use in each case). If the details aren't fully ironed out, you should do additional reasoning to do so before giving the answer .\n\n
            You'll need to reason in chains of thought in order to determine the transformation rule. Start your response by carefully reasoning in <reasoning></reasoning> tags.\n You follow a particular reasoning style. You break down complex problems into smaller parts and reason through them step by step, arriving at sub-conclusions before stating an overall conclusion. This reduces the extent to which you need to do large leaps of reasoning.\n\nYou reason in substantial detail for as long as is necessary to fully determine the transformation rule and resolve any ambiguities/uncertainties.
            The task itself in the ASCII, diff between input and output and array format is:{task}
            The original task in json format is {json_task}
            After you finished all your deliberations, skip two lines,write the sign \"=\" ten times, from one more new line write \"TRANSFORMATION RULE: \" and, from one more new line, state the transformation rule to which you arrived as a result of your deliberations again, followed by ten more \"=\" signs.
            """
            
            # Build interleaved content: prompt first, then all blocks in order
            content = [{"type": "text", "text": prompt}]
            
            for block in blocks:
                if block["type"] == "image_url" and isinstance(self.writer, anthropic.Anthropic):
                    url = block["image_url"]["url"]
                    base64_data = url.split(",")[1]
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_data,
                        }
                    })
                #else:
                #    content.append(block)
            
            desc = self.call_writer(content)
            logging.info(f"task description : {desc}")
            return desc

        descs = self._parallel_map(_one, items)
        return [d or "" for d in descs]

    

    def _selector_tag_prompt(self, task, taskprompt):
        """ARC tag prompt for the 'hemens' clusterer. Overridden per domain."""
        return f"""You are given a list of tasks in json format, task prompts, and possibly critique of unsuccessful attempts \
                    to solve them and the prompts of the tasks that are similar to them. \

                    The list of tasks is: {task}
                    the list of prompts(descritpions), and possibly critique and prompts of similar tasks: {taskprompt}
                    Categorize them in terms of 6 tags, picking the best description in the corresponding list for each tag.
                    objects : [rectangle, line, outline, irregular, overlapping, multicolor, diagonal-lines, grid-layouts]
                    transformations: [copy, layering, filling, rotation, flipping, translation, scaling, recolor, draw-lines]
                    procedures: [search, agentic-program, convolutional-program, for-each, alignment, ordering]
                    invariances: [scale, orientation, size, max-min, topology, simmetry]
                    concepts: [relative-position, containment, adjacency, counting]
                    other: [novel-properties, data-dependent-grid, multi-sample-mapping, pattern-completion]

                    Produce a json output named "hemens".
                    """

    def select_cluster(self, taskprompts, tasks):
        if self.selector == "hemens":
            nmcount = 0
            json_tasks = []
            try:
                for task, taskprompt in zip(tasks, taskprompts):
                    prompt = self._selector_tag_prompt(task, taskprompt)
                    text = self.call_writer(prompt)
                    match = re.search(r'"hemens"\s*:\s*\{', text)
                    if not match:
                        json_tasks.append(None)
                        nmcount += 1
                        continue
                    if nmcount >= 4:
                        print("No Hemens--based description of the tasks was produced. returning a random cluster ")
                        rng = np.random.default_rng()
                        random_integers = [rng.integers(0, self.num_tasks, size=n)]
                        return random_integers

                    start = match.end() - 1  # position of '{'

                    # Step 2: extract full JSON object via brace matching
                    depth = 0
                    for i in range(start, len(text)):
                        if text[i] == '{':
                            depth += 1
                        elif text[i] == '}':
                            depth -= 1
                            if depth == 0:
                                json_str = text[start:i+1]
                                break
                    else:
                        nmcount += 1
                        json_tasks.append(None)

                        continue
                    json_tasks.append(json.loads(json_str))
                    if nmcount >= 4:
                        print("No Hemens--based description of the tasks was produced. returning a random cluster ")
                        rng = np.random.default_rng()
                        random_integers = [rng.integers(0, self.num_tasks, size=n)]
                        return random_integers
            except Exception as e:
                print("Error in select_cluster ", e)

            
            sm = np.zeros((self.num_tasks, self.num_tasks))
            for i, itask in enumerate(json_tasks):
                for j, jtask in enumerate(json_tasks):
                    if i != j and itask is not None and jtask is not None:
                        #simsc = [itask[k] == jtask[k] for k in itask.keys()]
                        simsc = [itask[k] == jtask[k] for k in itask.keys() if k in jtask]

                        sm[i, j] = 1./sum(simsc) if sum(simsc) > 0 else 6
                    else:
                        sm[i, j] = 6
        else:
            tasklen = len(tasks)
            splt = tasklen/10
            response = None
            sm = None
            for j in range(0, tasklen, int(splt)):
                tasks_ = tasks[j:j+int(splt)]
                taskprompts_ = taskprompts[j:j+int(splt)]
                for attempt in range(3):
                    prompt = f"""You are given a list of tasks in json format, task prompts, and possibly critique of unsuccessful attempts \
                    to solve them and the prompts of the tasks that are similar to them. \
                    Please characterize the similarity between tasks. Formalize your characterization by creating a similarity distance \
                    matrix, where lower distance means more similar tasks, and the main diagonal is zeros, since it is the \
                    similarity distance between the task and itself. Write the result as a json object named "similarity_matrix".
                    The list of tasks is: {tasks_} 
                    the list of prompts, and possibly critique and prompts of similar tasks: {taskprompts_}
                    Tasks you previously clustered, if this is not the fisst batch, in form of the similarity matrix: {response}
                    Preprocessed simularity matrix from your previous output , if this is not the fisst batch: {sm}
                    Append to this similarity matrix"""
                    try:
                        content = self._complete("selector", prompt, 8192)
                    except BudgetExceeded:
                        raise
                    except Exception as e:
                        print("Error in select_cluster call:", e)
                        content = ""
                    match = re.search(r'\{.*?"similarity_matrix"\s*:\s*\[.*?\].*?\}', content, re.DOTALL)
                    if match:
                        break


           
           
            
                try:
                    obj = json_repair.loads(match.group(0))
                    sm = np.array(obj["similarity_matrix"])
                except Exception as e:
                    print("the model did not give the similarity matrix. returning a random cluster ")
                    print("OUR EXCEPTION , ", e)
                    print("obj  ", obj)
                    print("match  ", match)
                    print("content ", content)
                    rng = np.random.default_rng()
                    random_integers = [rng.integers(0, self.num_tasks, size=n)]
                    return random_integers
        
        
        # the logic is TO THE NEXT UNSOLVED TASK
        # 1) we select SOLVED tasks -- rows
        # 2) remove SOLVED tasks from columns


        # IT IS SERIOUS THAT BECAUSE OF THE SMALLER NUMBER OF PROMPTS THAT YOU APPEND,
        # YOU MAY GET WORSE THAN OTHERWISE RESULTS
    

        keep = np.array([not solved for solved in self.solved])
        sm_uncond = sm[~keep, :]
        sm = sm[np.ix_(~keep, keep)]
        


    


        # vectorized row stats (faster than your loop)
        row_mins  = sm.min(axis=1)
        row_means = sm.mean(axis=1)

        row_mins_uncond = sm_uncond.min(axis=1)
        row_means_uncond = sm_uncond.mean(axis=1)

        minind  = row_mins.argmin()
        meanind = row_means.argmin()

        # pick which compressed index to cluster around
        clustind = minind if row_mins[minind] < row_means[meanind] * p else meanind

    
        # get top-10 nearest unsolved tasks (in original task space)
        

        nearest = deque(np.where(keep)[0][sm[clustind].argsort()[:n-1]])
        nearest_uncond = deque(sm_uncond[clustind].argsort()[:n-1])
        cntr = 0
        j = 1
        for i in range(1, n-1):
            j += 1
            if self.solved[nearest_uncond[i]]:
                cntr += 1
            if cntr == 3:
                break
        k = 0
        for i in range(j, n-1):
            while k < n-1 and nearest[k] in nearest_uncond:
                k += 1
            if k >= len(nearest):
                break

            nearest_uncond[i] = nearest[k]
            k  += 1
        clustind = np.where(~keep)[0][clustind]
        nearest.appendleft(clustind)
        if clustind not in nearest_uncond:
            nearest_uncond.appendleft(clustind)
        return np.array(nearest_uncond)


    def write_code(self, prompt, task):
        prompt = f"""You are Guido van Rossum, a genius coder. Write a code for the following task, that defines the transformation: {task}. The transformation rule is delimited by 

        ========== 
        TRANSFORMATION RULE: 
        #here is the transformation rule
        ==========
        in the description below. YOU SHOULD READ the full description including any critiques and solved similar tasks before coding WHETHER YOU FIND THE delimiter or not.
        The description of the task, that is, the transformation and maybe the critique of the previous solutions, are as follows: """ + prompt + """. Your task is to write 
        the code, that is, the def transform(grid): function, maybe the comments, and nothing else. 
        Implement the transformation in code.\n You should write a function called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as list[list[int]]`). You should make sure that you implement a version of the transformation which works in general (for inputs which have the same properties as the example inputs .\n Don't write tests in your python code, just output the transform function.\n
        Write your code in triple backticks (```python and then ```). """
        logging.info(f"=====================================================")
        logging.info(f"Task {task} description that the coder gets: {prompt}")
        content = ''
        for _ in range(3):  # bounded; original `while not content` could spin forever
            try:
                content = self._complete("coder", prompt, 2048)
                if content:
                    logging.info(f"*****************************************************")
                    logging.info(f"code written: {content}")
                    logging.info(f"=====================================================")
                    return content
            except BudgetExceeded:
                raise
            except Exception as e:
                print("Error in call_coder:", e)
        logging.info(f"*****************************************************")
        logging.info(f"code written: {content}")
        logging.info(f"=====================================================")
        return content



    def run_code(self, code, task):
        #code = re.sub(r'```python\s*', '', code)
        #code = re.sub(r'```\s*', '', code)
        match = re.search(r'```python\s*(.*?)```', code, re.DOTALL)
        if match:
            code = match.group(1)
        else:
            #return [None] * len(task["train"])
            return [None] * len(task.train)

        namespace = {"np": np}
        try:
            exec(code, namespace)
        except Exception:
            #return [None] * len(task["train"])
            return [None] * len(task.train)
        outputs = []
        #for pair in task["train"]:
        for pair in task.train:
            try:
                #out = namespace["transform"](pair["input"])
                out = namespace["transform"](pair.input)
            except Exception:
                out = None
            outputs.append(out)
        return outputs

    def parse_output(self, output, task):
        #pairs = task["train"]
        pairs = task.train
        #correct = sum(out == pair["output"] for out, pair in zip(output, pairs) if out is not None)
        correct = sum(out == pair.output for out, pair in zip(output, pairs) if out is not None)
        return correct / len(pairs)
    
    
    def get_tasks(self):
        all_tasks = self.build_challenges_v2(Path(self.taskdir))
       
        
        taskids = list(all_tasks.keys())[:self.num_tasks]
        tasks = {k: all_tasks[k] for k in taskids}
        
        formatted_text = []
        formatted_blocks = []
        jsons = []
        for challenge in tasks.values():
            blocks = content_from_challenge(
                challenge=challenge,
                include_diffs=True,
                include_image=True,   # jetzt mit Bildern
                use_ascii=True,
                use_array=True,
            )
            text = "\n".join(b["text"] for b in blocks if b["type"] == "text")
            formatted_text.append(text)
            formatted_blocks.append(blocks)
            jsons.append(challenge)
        
        return tasks, formatted_text, formatted_blocks, taskids, jsons
        
        
        #j  = 0
        #for fname in sorted(os.listdir(self.taskdir)):
        #    if fname.endswith(".json"):
        #        with open(os.path.join(self.taskdir, fname)) as f:
        #            tasks.append(json.load(f))
        #        taskids.append(fname.removesuffix(".json"))
        #        j += 1
        #        if j == self.num_tasks:
        #            break
        # return tasks, formatted, taskids

    

    def run(self):
        tasks, formatted, formatted_blocks, taskids, jsons = self.get_tasks()
        print("got tasks ", len(tasks))
        print("TASK IDS ", taskids)
        logging.info(f"tasks formatted : {formatted}")

        # Phase 0: describe every task (parallel — see write_description).
        taskprompts = self.write_description(formatted, formatted_blocks, jsons)

        retry: dict = {}
        succeeded: dict = {}
        task_list = list(tasks.values())

        # ---------- Phase 1: initial solve, parallel across tasks ----------
        def _solve_initial(packed):
            idx, task, fmt, fb, prompt, json_task = packed
            code = self.write_code(prompt, fmt)
            output = self.run_code(code, task)
            score = self.parse_output(output, task)
            if score < 1:
                critique = self.write_self_critique(fmt, prompt, code, output, json_task)
                new_prompt = prompt + self._as_text(critique)
                return ("retry", idx,
                        (new_prompt, task, fmt, code, output, score, json_task, fb, critique))
            rule = self.extract_rule(prompt)
            return ("solved", idx, (prompt, task, fmt, code, json_task, rule, fb))

        initial_items = [
            (idx, task, fmt, fb, prompt, json_task)
            for idx, (task, fmt, fb, prompt, json_task) in enumerate(
                zip(task_list, formatted, formatted_blocks, taskprompts, jsons))
        ]
        for res in self._parallel_map(_solve_initial, initial_items):
            if res is None:
                continue
            kind, idx, payload = res
            if kind == "solved":
                succeeded[idx] = payload
                self.solved[idx] = True
                print("solved! ", idx)
            else:
                retry[idx] = payload

        n_solved = int(np.sum(self.solved))
        print("solved total: ", n_solved)
        if self.budget is not None:
            print(self.budget.summary())

        if n_solved > 0:
            idxs = self._cluster_or_random(taskprompts, formatted)
            newsolves = True
        else:
            idxs = np.random.randint(0, self.num_tasks, size=n)
            newsolves = False
        idxs = np.asarray(idxs).ravel().astype(int)

        # ---------- Phase 2: curriculum self-refinement retry rounds --------
        i = 0
        while i < self.num_retries:
            if self.budget is not None and self.budget.would_exceed():
                print("[budget] exhausted - stopping retry rounds.", self.budget.summary())
                break

            # Build the "successful similar tasks" context once per round.
            # compress_successful is cached per task (self._succ_compress_cache)
            # so we never pay to compress the same success twice.
            solved_in_cluster = [int(j) for j in idxs if self.solved[int(j)]]

            def _compress(j):
                with self._lock:
                    cached = self._succ_compress_cache.get(j)
                if cached is not None:
                    return (j, cached)
                out = self._as_text(self.compress_successful(succeeded[j][5], succeeded[j][3]))
                with self._lock:
                    self._succ_compress_cache[j] = out
                return (j, out)

            comp = {}
            for r in self._parallel_map(_compress, solved_in_cluster):
                if r is not None:
                    comp[r[0]] = r[1]

            succ_acc = ""
            succ_acc_codes = ""
            for j in solved_in_cluster:
                succ_acc += ("\nTASK NUMBER " + str(j) + " " + comp.get(j, "")
                             + "TASK ITSELF " + self._as_text(succeeded[j][2]))
                succ_acc_codes += ("\nCODE FOR TASK NUMBER " + str(j) + " "
                                   + self._as_text(succeeded[j][3]) + "\n")

            retry_idxs = [int(j) for j in idxs[1:]
                          if int(j) in retry and not self.solved[int(j)]]

            def _retry_one(j):
                prompt = retry[j][0]
                task = retry[j][1]
                fmt = retry[j][2]
                json_task = retry[j][6]
                fb = retry[j][7]
                critique = retry[j][8]
                # Append context.  The ORIGINAL code concatenated raw objects
                # here (prompt += task / fb / json_task), where task & json_task
                # are pydantic Challenge objects and fb is a list[dict] -> that
                # raises TypeError.  _as_text() coerces them.  See README_UPGRADE.md.
                prompt += self._as_text(task)
                prompt += self._as_text(fmt)
                prompt += self._as_text(fb)
                prompt += self._as_text(json_task)
                prompt += "this is the criticism of previous unsuccessful attempts:" + self._as_text(critique)
                prompt += "this is the description of similar tasks that you have successfully solved:" + succ_acc
                prompt += "this is the code you have written for these tasks that you have successfully solved:" + succ_acc_codes
                code = self.write_code(prompt, fmt)
                output = self.run_code(code, task)
                score = self.parse_output(output, task)
                if score < 1:
                    critique2 = self.write_self_critique(fmt, prompt, code, output, json_task)
                    new_prompt = prompt + self._as_text(critique2)
                    return ("retry", j,
                            (new_prompt, task, fmt, code, output, score, json_task, fb, critique2))
                rule = self.extract_rule(prompt)
                return ("solved", j, (prompt, task, fmt, code, json_task, rule, fb))

            for res in self._parallel_map(_retry_one, retry_idxs):
                if res is None:
                    continue
                kind, j, payload = res
                if kind == "solved":
                    succeeded[j] = payload
                    self.solved[j] = True
                    print("resolved idx=", j)
                else:
                    retry[j] = payload
                    taskprompts[j] = payload[0]
                    print("not resolved idx=", j)

            if np.sum(self.solved) == self.num_tasks:
                break
            i += 1
            if int(np.sum(self.solved)) > n_solved:
                idxs = self._cluster_or_random(taskprompts, formatted)
                idxs = np.asarray(idxs).ravel().astype(int)
                print("new cluster ", idxs)
                newsolves = True
                n_solved = int(np.sum(self.solved))
            else:
                print("nothing was resolved this round")
                if n_solved == 0:
                    idxs = np.random.randint(0, self.num_tasks, size=n)
                    idxs = np.asarray(idxs).ravel().astype(int)
                    newsolves = False
                else:
                    continue

        if self.budget is not None:
            print(self.budget.summary())
        result = {"solved": int(np.sum(self.solved)), "num_tasks": self.num_tasks}
        print("FINAL:", result)
        return result


class MathDomainMixin:
    """Retarget the ARC/grid operations onto AIME-style math problems.

    Compose with an Orchestrator (see MathOrchestrator). Everything else —
    multithreading, budget, curriculum clustering, critique, compression — is
    inherited unchanged. The gold answer is never serialised into any prompt
    (see _as_text / write_self_critique) so the model can't cheat.
    """

    # set on the instance by main() after construction
    dataset_source: str = "aime"
    dataset_path = None

    @staticmethod
    def _as_text(x) -> str:
        # Strip the gold answer when a MathTask is folded into a prompt.
        from math_dataset import MathTask
        if isinstance(x, MathTask):
            return json.dumps({"id": x.id, "problem": x.problem})
        return Orchestrator._as_text(x)

    def get_tasks(self):
        from math_dataset import load_math_tasks
        src = self.dataset_path or self.dataset_source
        items = load_math_tasks(src, self.num_tasks)
        if not items:
            raise SystemExit(f"no math tasks loaded from {src!r}")
        # resize bookkeeping to however many actually loaded
        self.num_tasks = len(items)
        self.solved = [False] * self.num_tasks
        tasks = {t.id: t for t in items}
        taskids = list(tasks.keys())
        formatted = [t.problem for t in items]   # plain problem text
        formatted_blocks = [[] for _ in items]   # no images
        jsons = items                            # MathTask objects
        print(f"loaded {len(items)} math tasks from {src!r}")
        return tasks, formatted, formatted_blocks, taskids, jsons

    def write_description(self, tasks, formatted_blocks, jsons):
        items = list(zip(tasks, jsons))

        def _one(it):
            problem_text, _task = it
            prompt = (
                "You are an expert competition mathematician. Analyse the "
                "problem below and produce a rigorous SOLUTION PLAN (do NOT "
                "compute the final number yet): name the area (number theory, "
                "algebra, combinatorics, geometry, ...), the key theorems / "
                "techniques, and the main steps.\n\nPROBLEM:\n"
                + self._as_text(problem_text)
            )
            desc = self.call_writer(prompt)
            logging.info(f"math approach: {desc}")
            return desc

        return [d or "" for d in self._parallel_map(_one, items)]

    def write_code(self, prompt, task):
        # `task` is the problem text (fmt); `prompt` is the plan (+ critiques +
        # similar solved problems accumulated by run()).
        solve_prompt = (
            "You are an expert competition mathematician solving an AIME-style "
            "problem whose answer is a single integer (0-999 for AIME). Using "
            "the analysis and any prior attempts / similar solved problems "
            "below, reason step by step, then give ONLY the final answer inside "
            "\\boxed{ }.\n\nPROBLEM:\n" + self._as_text(task)
            + "\n\nANALYSIS / PRIOR ATTEMPTS / SIMILAR SOLVED PROBLEMS:\n"
            + self._as_text(prompt)
            + "\n\nEnd with the final answer as \\boxed{<integer>}."
        )
        content = ""
        for _ in range(3):
            try:
                content = self._complete("coder", solve_prompt, 4096)
                if content:
                    logging.info(f"math solution: {content}")
                    return content
            except BudgetExceeded:
                raise
            except Exception as e:
                print("Error in math solve:", e)
        return content

    def run_code(self, code, task):
        # "code" is the solution text; the checkable form is the extracted answer.
        from math_dataset import extract_final_answer
        return extract_final_answer(code or "")

    def parse_output(self, output, task):
        from math_dataset import grade_answer
        gold = getattr(task, "answer", None)
        if gold is None and isinstance(task, dict):
            gold = task.get("answer")
        return 1.0 if grade_answer(output or "", gold or "") else 0.0

    def extract_rule(self, desc):
        # stored as the 'rule' for a solved task, then fed to compress_successful.
        return desc

    def write_self_critique(self, task, prompt, code, output, json_task):
        critique_prompt = (
            "A competition math problem was attempted unsuccessfully. Critique "
            "the attempt: where is the reasoning wrong and what should be tried "
            "instead? Do NOT reveal or guess the official answer.\n"
            "PROBLEM:\n" + self._as_text(task)
            + "\nPRIOR PLAN / CRITIQUES:\n" + self._as_text(prompt)
            + "\nLAST ATTEMPT (full solution):\n" + self._as_text(code)
            + "\nITS EXTRACTED ANSWER: " + self._as_text(output)
            + "\nGive a concise critique (<= 300 words)."
        )
        critique = self.call_writer(critique_prompt, max_tokens=2048)
        logging.info(f"math critique: {critique}")
        return critique

    def _selector_tag_prompt(self, task, taskprompt):
        return f"""You are given a competition math problem (json) and its solution plan / critique.
                    The problem: {task}
                    The plan / critique / similar problems: {taskprompt}
                    Categorize it using these tag groups, picking the best-matching tag(s) per group.
                    area: [number-theory, algebra, combinatorics, geometry, probability, sequences, functional-equations]
                    techniques: [modular-arithmetic, generating-functions, recursion, casework, invariants, inequalities, pigeonhole, complex-numbers, coordinate-geometry, vieta]
                    objects: [integers, polynomials, primes, divisors, points, triangles, circles, sets, permutations]
                    structure: [counting, optimization, existence, construction, evaluation]
                    difficulty: [computational, insight-heavy, multi-step]
                    Produce a json output named "hemens".
                    """


class MathOrchestrator(MathDomainMixin, Orchestrator):
    """The ARC orchestrator's machinery, retargeted to math problems."""
    pass


def _env_key_for(model: str):
    """Pick the right API key env var for a model name."""
    if model[:3] == "gpt" or model[:2] in ("o1", "o3", "o4"):
        return os.environ.get("OPENAI_API_KEY")
    if model[:4] in ["haik", "sonn", "opus", "anth", "clau"]:
        return os.environ.get("ANTHROPIC_API_KEY")
    return os.environ.get("FIREWORKS_API_KEY")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="ARC-AGI-2 evolutionary self-refinement orchestrator "
                    "(multithreaded, USD-budget-aware).")
    ap.add_argument("--writer", default="claude-haiku-4-5-20251001")
    ap.add_argument("--coder", default="claude-haiku-4-5-20251001")
    ap.add_argument("--selector", default="hemens",
                    help='"hemens" (tag-based, default) or a model id.')
    ap.add_argument("--dataset", default="arc", choices=["arc", "aime"],
                    help="task domain: 'arc' (ARC-AGI-2 grids, default) or "
                         "'aime' (AIME competition math, exact integer answers).")
    ap.add_argument("--dataset-path", default=None,
                    help="math only: local .json/.jsonl file OR a HF dataset id "
                         "(e.g. AI-MO/aimo-validation-aime); overrides the "
                         "default source. Use 'aime-sample' for the offline sample.")
    ap.add_argument("--num-tasks", type=int, default=40)
    ap.add_argument("--num-trials", type=int, default=None,
                    help="retry rounds (maps to num_retries). If omitted while "
                         "--budget-usd is set, it is derived from the budget.")
    ap.add_argument("--context-length", "--max-context", dest="context_length",
                    type=int, default=None,
                    help="max generated (output) tokens per call. If omitted "
                         "while --budget-usd is set, derived from the budget.")
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="total USD budget. Derives num-trials / context-length "
                         "when they are not given, and is enforced at runtime.")
    ap.add_argument("--workers", type=int, default=8,
                    help="max concurrent LLM calls (threads).")
    ap.add_argument("--price-per-1m-tokens", type=float, default=None,
                    help="override blended $/1M-tokens used for budget planning.")
    ap.add_argument("--price-file", default=None,
                    help="JSON {model: [in_per_mtok, out_per_mtok]} to merge into the table.")
    ap.add_argument("--io-input-weight", type=float, default=0.5,
                    help="fraction of a token-unit treated as input when blending "
                         "price for planning (0..1; ARC prompts are input-heavy).")
    ap.add_argument("--yes", action="store_true",
                    help="auto-accept the balanced-budget proposal (non-interactive).")
    ap.add_argument("--api-key-writer", default=None)
    ap.add_argument("--api-key-coder", default=None)
    ap.add_argument("--api-key-selector", default=None)
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    if args.price_file:
        load_price_file(args.price_file)

    key_w = args.api_key_writer or _env_key_for(args.writer)
    key_c = args.api_key_coder or _env_key_for(args.coder)
    if args.selector == "hemens":
        key_s = "hemens"
    else:
        key_s = args.api_key_selector or _env_key_for(args.selector)

    for name, key in (("writer", key_w), ("coder", key_c)):
        if not key:
            raise SystemExit(
                f"No API key for {name}. Set ANTHROPIC_API_KEY / OPENAI_API_KEY / "
                f"FIREWORKS_API_KEY in the environment, or pass --api-key-{name}.")

    # ---- budget planning: budget USD -> (num_trials, context_length) ----
    if args.price_per_1m_tokens is not None:
        blended = args.price_per_1m_tokens
    else:
        blended = blended_price_per_mtok(args.writer, input_weight=args.io_input_weight)

    plan = resolve_budget(
        budget_usd=args.budget_usd,
        num_trials=args.num_trials,
        context_length=args.context_length,
        num_tasks=args.num_tasks,
        blended_mtok=blended,
        assume_yes=args.yes,
    )
    for note in plan.notes:
        print("[budget]", note)
    print(f"[budget] effective: num_trials={plan.num_trials} "
          f"context_length={plan.context_length} "
          f"budget_enabled={plan.budget_enabled} "
          f"max_usd={plan.max_usd}")

    budget = BudgetTracker(max_usd=plan.max_usd) if plan.budget_enabled else None

    common = dict(
        num_tasks=args.num_tasks,
        num_retries=plan.num_trials,
        max_workers=args.workers,
        max_output_tokens=plan.context_length,
        budget=budget,
    )
    if args.dataset == "arc":
        orchestrator = Orchestrator(
            args.writer, args.coder, args.selector, key_w, key_c, key_s, **common)
    else:
        orchestrator = MathOrchestrator(
            args.writer, args.coder, args.selector, key_w, key_c, key_s, **common)
        orchestrator.dataset_source = args.dataset        # e.g. "aime"
        orchestrator.dataset_path = args.dataset_path     # optional override
    print(f"[dataset] {args.dataset}"
          + (f" (source={args.dataset_path})" if args.dataset_path else ""))
    orchestrator.run()


if __name__ == "__main__":
    main()
