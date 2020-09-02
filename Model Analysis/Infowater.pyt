import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import arcpy
import sys
import re

import pandas as pd
import numpy as np

import os
import re

import argparse

from collections import OrderedDict

import datetime

try:
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.utils.cell import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo
    from openpyxl import Workbook, load_workbook, worksheet
    op_xl = True
except:
    op_xl = False
    arcpy.AddMessage("No openpyxl")

#model comparison imports
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from arcgis.features import GeoAccessor, GeoSeriesAccessor

class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Infowater toolbox"
        self.alias = ""

        # List of tool classes associated with this toolbox
        self.tools = [AnalyzeModelReport, ConvertModelReport, ModelComparison]


class AnalyzeModelReport(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Analyze Model Report"
        self.description = "Tool to analyze the output of a model"
        self.canRunInBackground = False
        self.category = "Model Report Analysis"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
            displayName="Model Output File",
            name="model_output_file",
            datatype="DEFile",
            parameterType="Required",
            direction="Input",
        )
        param0.filter.list = ["RPT"]

        param1 = arcpy.Parameter(
            displayName="Output Folder",
            name="output_folder",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input",
        )
        param1.filter.list = ["File System"]

        param2 = arcpy.Parameter(
            displayName="Output Excel Name",
            name="output_excel_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        # param2.filter.list = []

        param3 = arcpy.Parameter(
            displayName="Pressure Zone Sheet",
            name="zone_sheet",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input",
        )
        param3.filter.list = ['xls', 'xlsx']
        params = [param0, param1, param2, param3]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""

        # test for py3
        if sys.version_info.major != 3:
            arcpy.AddWarning("Need to run in ArcGIS Pro")
            raise
    

        file_path = parameters[0].valueAsText
        output_folder= parameters[1].valueAsText
        output_name= parameters[2].valueAsText
        zone_sheet= parameters[3].valueAsText

        self.do_work(file_path, output_folder, output_name, zone_sheet)
        

    def do_work(self, file_path, output_folder, output_name, zone_sheet):

        if output_folder is None:
            output_folder = ''

        if len(re.findall('\.xlsx$', output_name, re.IGNORECASE)) == 0:
               output_name += ".xlsx"

        with open(
                file_path, 'r') as fl:
            data = fl.read()
        txt = data.split("\n")
        txt = txt

        trials_dataframes = []
        warning_dataframes = []
        change_dataframes = []
        balance_dataframes = []

        time_group = self._generate_time_groups(self.create_search_text(txt))

        arcpy.AddMessage(f"Processing {len(time_group)} time groups")
        for gs in time_group:
            
            time_s = self.determine_timestamp(gs)
            nw, w = self.split_warnings(gs)
            trials_dataframes.append(self._generate_trials_dataframe(nw, time_s))
            warning_dataframes.append(self._generate_warning_dataframe(w, time_s))
            change_dataframes.append(self._generate_change_dataframe(nw, time_s))
            balance_dataframes.append(self._generate_complete_trial_dataframe(nw, time_s))

        warning_df = pd.concat(warning_dataframes, ignore_index=True)
        trial_df = pd.concat(trials_dataframes, ignore_index=True)
        change_df = pd.concat(change_dataframes, ignore_index=True)
        balance_df = pd.concat(balance_dataframes, ignore_index=True)

        warning_df = self.format_df_to_time(warning_df)
        trial_df = self.format_df_to_time(trial_df)
        change_df = self.format_df_to_time(change_df)
        balance_df = self.format_df_to_time(balance_df)

        if zone_sheet is not None:
            arcpy.AddMessage("Processing Pressure Zone Join")
            _zone_data = pd.read_excel(zone_sheet)
            warning_df = self._merge_zones(warning_df, _zone_data)
            trial_df = self._merge_zones(trial_df, _zone_data)
            change_df = self._merge_zones(change_df, _zone_data)

        arcpy.AddMessage("Exporting Data to excel")
        arcpy.AddMessage(f"Folder: {output_folder}")
        arcpy.AddMessage(f'File: {output_name}')
        if op_xl:
            self._export_to_excel_tables(frame_list=[
                ["Changes", change_df],
                ["Trials", trial_df],
                ["Warnings", warning_df],
                ["Balance", balance_df],
            ], output_folder=output_folder, output_name=output_name)
        else:
            self._export_to_excel(frame_list=[
                ["Changes", change_df],
                ["Trials", trial_df],
                ["Warnings", warning_df],
                ["Balance", balance_df],
            ], output_folder=output_folder, output_name=output_name)
            
        return


    def create_search_text(self, _text):
        hs_line = 0
        for i, l in enumerate(_text):
            if "Hydraulic Status" in l:
                hs_line = i
        return _text[hs_line+2:]


    def _generate_time_groups(self, _text):
        time_group = []
        ws_count = 0
        line_list = []
        for t in _text:
            out_t = re.sub(pattern='^[ \t]+|[ \t]+$', string=t, repl='')
            if re.match(pattern='^\s+$', string=t) is not None:
                # new lines are generally split by the whitespace then warnings
                ws_count += 1
                continue
            elif (re.match(pattern='^\s+\d{1,2}:\d{1,2}:\d{1,2}', string=t) is not None) and (ws_count > 0):
                # this indicates a new line
                ws_count = 0
                time_group.append(line_list)
                line_list = []
            elif ws_count >= 2:
                continue
            line_list.append(out_t)
        time_group.append(line_list)
        return time_group


    def _element_name(self, value):
        mt = re.findall(string=value, pattern='\w*_{1}\w*')
        if len(mt) > 0:
            mt = mt[0]
        else:
            mt = None

        return str(mt)


    # Trials
    def get_net_balance_rows(self, _list_values):
        start_row = 0
        end_row = 9999
        for i, item in enumerate(_list_values):
            if "Balancing the network".upper() in item.upper():
                start_row = i+1
            elif "Balanced".upper() in item.upper():
                end_row = i
        return start_row, end_row


    def get_trial_list(self, _list_items):
        trials = []
        for i, item in enumerate(_list_items):
            if "Trial" in item:
                trials.append(i)
        trials.append(None)
        return trials


    def _get_trials_pairs(self, _trials_list):
        pairs = []
        for i in range(0, len(_trials_list)-1):
            pairs.append([_trials_list[i], _trials_list[i+1]])
        return pairs


    def _generate_trials_output(self, _list_items, _pairs, ts=None):
        out_trials = []
        for i, item in enumerate(_pairs):
            its = _list_items[item[0]+1:item[1]]
            if len(its) == 0:
                out_trials.append([ts, i+1, None, None])
            else:
                for l in _list_items[item[0]+1:item[1]]:
                    mt = self._element_name(l)
                    row = [ts, i+1, mt, l]
                    out_trials.append(row)
        return out_trials


    def _generate_trials_dataframe(self, _list_values, _ts=None):
        bal_s, bal_e = self.get_net_balance_rows(_list_values)
        t_list = self.get_trial_list(_list_values[bal_s:bal_e])
        t_pairs = self._get_trials_pairs(t_list)
        t_out = self._generate_trials_output(_list_values[bal_s:bal_e], t_pairs, _ts)
        t_df = pd.DataFrame(data=t_out, columns=["Time", "Trial", "Element", "Details"])
        return t_df


    def _generate_warning_dataframe(self, _list_values, _ts=None):
        warnings = []
        for row in _list_values:
            row = row.replace("WARNING: ", "")
            element = str(self._element_name(row))
            warning_t = self._warning_type(row)
            warnings.append([_ts, warning_t, element, row])
        warnings_df = pd.DataFrame(data=warnings, columns=["Time", "Warning_Type", "Element", "Details"])
        return warnings_df


    def _warning_type(self, value):
        value = value.upper()
        if "cannot deliver flow".upper() in value:
            er = "FLOW"
        elif "inability to deliver head".upper() in value:
            er = "HEAD"
        elif "negative pressures at demand node".upper() in value:
            er = "NEGATIVE PRESSURE"
        else:
            er = "OTHER"
        return er


    def _generate_change_dataframe(self, _list_values, _ts=None):
        change_list = []
        for item in _list_values:
            match = re.match(pattern='(\d{1,2}:\d{2}:\d{2})(?!: Balanc[e|ing])',
                             string=item)
            if match is None:
                continue

            out_item = re.sub(pattern='\d{1,2}:\d{2}:\d{2}:\s*', string=item,
                              repl='')
            element = self._element_name(out_item)
            change_list.append([_ts, element, out_item])

        change_df = pd.DataFrame(data=change_list, columns=["Time", "Element", "Details"])
        return change_df


    def format_df_to_time(self, _df):
        _df["Hour"] = _df["Time"].apply(lambda x: int(x.split(":")[0]))
        _df["Minute"] = _df["Time"].apply(lambda x: int(x.split(":")[1]))
        _df["Second"] = _df["Time"].apply(lambda x: int(x.split(":")[2]))

        _df["Timestep"] = (_df["Hour"] / 1) + (_df["Minute"] / 60) + (
                    (_df["Second"] / 60) / 60)
        _df["Timestep"] = _df["Timestep"].round(4)
        return _df


    def _generate_complete_trial_dataframe(self, _list_values, _ts=None):
        t_pat = re.compile("(?<=Balanced after )\d*(?= trials)")
        s_pat = re.compile("(?<=Total Supplied:\s\s)\d*.\d*(?=\w*)")
        d_pat = re.compile("(?<=Total Demanded:\s\s)\d*.\d*(?=\w*)")
        e_pat = re.compile("(?<=Total Stored:\s\s\s\s)\d*.\d*(?=\w*)")
        trials = self._get_trial_pattern_data(_list_values, t_pat)
        supplied = self._get_trial_pattern_data(_list_values, s_pat)
        demanded = self._get_trial_pattern_data(_list_values, d_pat)
        stored = self._get_trial_pattern_data(_list_values, e_pat)

        _df = pd.DataFrame(data=[[_ts, trials, supplied, demanded, stored]],
                           columns=["Time", "Trials", "Supplied", "Demanded",
                                    "Stored"])
        return _df


    def _get_trial_pattern_data(self, _list_values, pattern):
        data = [float(y) for x in _list_values for y in pattern.findall(string=x) if len(y) > 0]
        if len(data) > 0:
            return data[0]
        else:
            return None


    def determine_timestamp(self, _list_values):
        for item in _list_values:
            mt = re.findall(pattern='^\s*(\d{1,2}:\d{1,2}:\d{1,2})', string=item)
            if mt is None:
                continue
            return mt[0]


    def split_warnings(self, _list_values):
        non_warnings = []
        warnings = []
        for item in _list_values:
            if "WARNING" in item.upper():
                warnings.append(item)
            else:
                non_warnings.append(item)

        return non_warnings, warnings

    # region Excel

    def _export_to_excel(self, frame_list, output_folder, output_name=None):
        if output_name is None:
            output_name = "Model_Report_Analysis.xlsx"

        if not output_name.endswith(".xlsx"):
            output_name = "{}.xlsx".format(output_name)

        print("Saving to {}".format(os.path.join(output_folder, output_name)))

        with pd.ExcelWriter(os.path.join(output_folder, output_name)) as writer:
            for i in frame_list:
                i[1].to_excel(writer, sheet_name=i[0])


    def _export_to_excel_tables(self, frame_list, output_folder, output_name):
        if output_name is None:
            output_name = "Model_Report_Analysis.xlsx"

        if not output_name.endswith(".xlsx"):
            output_name = "{}.xlsx".format(output_name)

        print("Saving to {}".format(os.path.join(output_folder, output_name)))

        wb = Workbook()

        for i in frame_list:
            frame = i[1]
            name = i[0]
            print("Exporting Sheet {}".format(name))
            self._frame_to_table(frame, name, wb)

        wb.remove(wb[wb.sheetnames[0]])
        print("Saving")
        wb.save(os.path.join(output_folder, output_name))


    def _frame_to_table(self, frame, name, wb):
        ws = wb.create_sheet()
        ws.title = name

        row_min = 1
        row_max = frame.shape[0] + 1

        col_min = get_column_letter(1)
        col_max = get_column_letter(len(frame.columns))

        tbl_idx = "{}{}:{}{}".format(col_min, row_min, col_max, row_max)

        ws.append(list(frame.columns))
        for row in dataframe_to_rows(frame, index=False, header=False):
            ws.append(row)

    
        # if the data frame is empty just return
        if frame.empty:
            return

        tab = Table(displayName='{}_tbl'.format(name), ref=tbl_idx)
        style = tab.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2", showFirstColumn=False,
            showLastColumn=False, showRowStripes=True
        )
        tab.tableStyleInfo = style
        ws.add_table(tab)

        self._format_column_width(frame, ws)


    def _format_column_width(self, frame, ws):
        measurer = np.vectorize(len)
        widths = [max(8, max(x)) + 2 for x in list(zip(measurer(frame.columns), measurer(frame.values.astype(str)).max(axis=0)))]
        # print(widths)
        for i, column_width in enumerate(widths):
            ws.column_dimensions[get_column_letter(i + 1)].width = column_width
    # endregion

    def _merge_zones(self, _df, _zone_df):
        out_df = _df.merge(
            how='left', left_on='Element',
            right_on='ID', right=_zone_df
        )
        return out_df.drop("ID", axis=1)



class ConvertModelReport(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Convert Model Report Excel"
        self.description = "Convert the model report file from excel"
        self.canRunInBackground = False
        self.category = "Model Report Analysis"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
            displayName="Output Excel Document",
            name="excel_doc",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input",
        )
        param1 = arcpy.Parameter(
            displayName="Output Geodatabase",
            name="output_gdb",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input",
        )

        param2 = arcpy.Parameter(
            displayName="Table Name",
            name="tbl_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        
        param0.filter.list = ['xls', 'xlsx']
        param1.filter.list = ['Local Database']
        
        params = [param0, param1, param2]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        excel_doc = parameters[0].valueAsText
        out_gdb = parameters[1].valueAsText
        base_name = parameters[2].valueAsText
        
        for sht in ["Changes", "Trials", "Warnings", "Balance"]:
            arcpy.AddMessage(f"Exporting Sheet: {sht}")
            tbl_name = f"{base_name}_{sht}"
            tbl_path = os.path.join(out_gdb, tbl_name)
            arcpy.conversion.ExcelToTable(excel_doc, tbl_path, sht)
        
        return        


#---------------------------------

class ModelComparison(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Model Comparison"
        self.description = "Generate PDF outputs of two models to compare"
        self.canRunInBackground = False
        
        try:
            self.crs = matplotlib.cm.get_cmap('tab10')
            self.cls = dict(zip(["NewData", "OldData", "SCADA"], [self.crs(x) for x in np.linspace(0, 0.2, 3)]))
        except:
            print("exception on colors")
            self.crs = [(0.267004, 0.004874, 0.329415, 1.0), (0.172719, 0.448791, 0.557885, 1.0), (0.369214, 0.788888, 0.382914, 1.0)]
            self.cls = dict(zip(["NewData", "OldData", "SCADA"], self.crs))
            # plt.style.use("bmh")

        self.new_scenario_data = None
        self.old_scenario_data = None
        self.comparison_table = None
        self.sc = None

        self.hs = 0.3
        self.ws = 0.12

        self.bbox_x = -0.5
        self.bbox_y = 1.1

        self.poi_lim_mod = 2

        self.new_model_label = None
        self.old_model_label = None

               

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
            displayName="Old Model Scenario",
            name="old_model_scenario",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input",
        )
        
        param1 = arcpy.Parameter(
            displayName="New Model Scenario",
            name="new_model_scenario",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input",
        )

        param2 = arcpy.Parameter(
            displayName="Comparison Sheet",
            name="comparison_sheet",
            datatype="DEFile",
            parameterType="Required",
            direction="Input"
        )
        param3 = arcpy.Parameter(
            displayName="PDF Output Folder",
            name="pdf_folder",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input"
        )
        
        param4 = arcpy.Parameter(
            displayName="Sub Zone List",
            name="zone_list",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
            multiValue=True
        )
        param5 = arcpy.Parameter(
            displayName="SCADA Data",
            name="scada_data",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input"
        )

        param0.filter.list = ['File System']
        param1.filter.list = ['File System']
        param2.filter.list = ['xls', 'xlsx']
        param3.filter.list = ['File System']
        param5.filter.list = ['xls', 'xlsx']
        
        params = [param0, param1, param2, param3, param4, param5]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""

        # [arcpy.AddMessage(str(i.valueAsText)) for i in parameters]

        _old_model = parameters[0].valueAsText
        _new_model = parameters[1].valueAsText
        _comp_sheet = parameters[2].valueAsText
        _pdf_f = parameters[3].valueAsText        
        _zone = parameters[4].valueAsText
        _scada = parameters[5].valueAsText
        
        self.do_work(_new_model, _old_model, _comp_sheet, _scada, _zone, _pdf_f)
        return

    def do_work(self, new_model, old_model, comparison_sheet, scada, zone, pdf_folder, new_model_label=None, old_model_label=None):
        _start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        arcpy.AddMessage(_start_time)

        self.new_model_label = "New Model" if new_model_label is None else new_model_label
        self.old_model_label = "Old Model" if old_model_label is None else old_model_label
        
        arcpy.env.overwriteOutput=True
        _new_scenario_data = self.load_model_data(new_model)
        _old_scenario_data = self.load_model_data(old_model)
        _comparison_table = self.load_comparison_file(comparison_sheet)
        _sc = self.load_scada(scada)

        zs = set(_comparison_table['Pressure_Zone'].values)

        # zs = zones
        if zone is not None:
            zs = [i.replace("'", "") for i in zone.split(";")]

        self.new_scenario_data = _new_scenario_data
        self.old_scenario_data = _old_scenario_data
        self.comparison_table = _comparison_table
        self.sc = _sc

        if not os.path.isdir(pdf_folder):
            os.makedirs(pdf_folder)

        t_count = len(zs)
        arcpy.SetProgressor("step", "Looping pressure zones", 0, t_count, 1)
        
        for i_iteration, zone in enumerate(zs):
            arcpy.AddMessage("Zone - {}".format(zone))
            arcpy.SetProgressorPosition(i_iteration)
            arcpy.SetProgressorLabel("Zone - {}".format(zone))
            with PdfPages(os.path.join(pdf_folder, '{}.pdf'.format(zone))) as pp:
                self.zone_base(self.comparison_table.loc[self.comparison_table["Pressure_Zone"] == zone], pp)
        arcpy.AddMessage("Done")

    def load_comparison_file(self, file_path):
        return pd.read_excel(file_path)

    def load_scada(self, scada_path):
        try:
            if ".csv" in scada_path:
                df = pd.read_csv(scada_path)
                fm = '%Y-%m-%d %H:%M:%S'
            else: 
                df = pd.read_excel(scada_path)
                fm = "%m/%d/%Y %H:%M:%S"

            df.head()
            df['Time'] = pd.to_datetime(df['Time'], format=fm)
            df = df.set_index('Time')

            df['Minute'] = df.index.minute
            df['Hour'] = df.index.hour

            return df
        # except:
        #     pass
        # try:
        #     self.sc = pd.read_excel(scada_path)
        #     self.sc["Time"] = self.sc["Time"].apply(lambda x: x+1)
        #     self.sc = self.sc.set_index("Time")
        #     return self.sc
        except:
            return None

    def load_dbf(self, scenario_path, dbf_name):
        try:
            # updated logic for arcigs pro. doesnt required simpledbf package this way
            _in_table = os.path.join(scenario_path, dbf_name)
            _out_table = dbf_name
            _out_table = _out_table.replace(".dbf", "")

            mem_table = arcpy.conversion.TableToTable(_in_table, "in_memory", f"{_out_table}_table")[0]
            df = pd.DataFrame.spatial.from_table(mem_table)
            return df
        except Exception as e:
            arcpy.AddError(e)
            return None

    def load_model_data(self, scenario):
        jct_data = self.load_dbf(scenario, "JunctOut.dbf")
        ppe_data = self.load_dbf(scenario, "PipeOut.dbf")
        tank_data = self.load_dbf(scenario, "TankOut.dbf")
        return {"junctions": jct_data, "pipes": ppe_data, "tanks": tank_data}

    # region Fill with JP

    def POI_chart(self, poi_data, pdf_f=None):
        poi_data = poi_data[["Old_Id", "New_Id", "Info"]]

        def chunks(l, n):
            for i in range(0, len(l), n):
                yield l[i:i + n]

        pl = self._get_poi_lims(poi_data)

        for vals in list(chunks(poi_data, 4)):
            self._generate_poi_subcharts(vals, pdf_f, pl)

    def _get_poi_lims(self, poi_data):
        od = self.old_scenario_data['junctions']
        nd = self.new_scenario_data['junctions']
        ov = poi_data["Old_Id"].values
        mx = max(od.loc[od["ID"].isin(ov)]["HEAD"].max(),
                 nd.loc[nd["ID"].isin(poi_data["New_Id"])]["HEAD"].max())
        mn = min(od.loc[od["ID"].isin(ov)]["HEAD"].min(),
                 nd.loc[nd["ID"].isin(poi_data["New_Id"])]["HEAD"].min())
        lim_mod = self.poi_lim_mod

        if any([np.isnan(mn), np.isnan(mx)]):
            return None
        else:
            return (mn - lim_mod, mx + lim_mod)

    def _generate_poi_subcharts(self, poi_list, pdf_f=None, poi_lims=None):

        fig = plt.figure()
        fig.set_size_inches(17, 11)
        fig.suptitle("Points of Interest")

        gr = plt.GridSpec(2, 2, wspace=self.ws, hspace=self.hs)

        index_list = [[0, 0], [0, 1], [1, 0], [1, 1]]

        list_items = [
            [
                "New_Id", self.cls["NewData"],
                "New Model", self.new_scenario_data['junctions']
            ],
            [
                "Old_Id", self.cls["OldData"],
                "Old Model", self.old_scenario_data['junctions']
            ] 
        ]

        for item, gr_index in zip(poi_list.iterrows(), index_list):
            ax = fig.add_subplot(gr[gr_index[0], gr_index[1]])
            idx, values = item

            for i in list_items:
                if not pd.isnull(values[i[0]]):
                    self._plot_poi(
                        ax, values[i[0]],
                        i[1], i[2], i[3]
                    )
                    
            # ax.legend()
            ax.grid()
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Head (m)")
            if not pd.isnull(values[2]):
                ax.set_title(values[2])

            if poi_lims is not None:
                ax.set_ylim(poi_lims)
            else:
                self.test_lims(ax)

        self.generate_legend(fig)

        if pdf_f is not None:
            pdf_f.savefig()
            plt.close()

    def _plot_poi(self, ax, _id, color, label, data):
        plot_data = data.loc[data["ID"] == _id].set_index("TIME_STEP")
        ax.plot(plot_data["HEAD"], label=label, color=color)

        self.plot_daily_avg(plot_data["HEAD"], ax, color, label)

        ax.set_xlim((1, 24))
        ax.set_xticks(list(range(1, 25)))

    def station_without_tank_graph(self, junction, pipe, site_name=None, pdf_f=None):
        fig = plt.figure()
        fig.set_size_inches(17, 11)
        if site_name is not None:
            fig.suptitle("Site - {}".format(site_name))

        gr = plt.GridSpec(1, 2, hspace=self.hs)
        a_jct = fig.add_subplot(gr[0, 0])
        a_ppe = fig.add_subplot(gr[0, 1])

        self.plot_junction(junction, a_jct)
        self.plot_pipe(pipe, a_ppe)

        self.generate_legend(fig)

        if pdf_f is not None:
            pdf_f.savefig()
            plt.close()

    def station_with_tank_graph(self, junction, pipe, tank, site_name=None, pdf_f=None):
        fig = plt.figure()
        fig.set_size_inches(17, 11)
        if site_name is not None:
            fig.suptitle("Site - {}".format(site_name))

        gr = plt.GridSpec(2, 2, wspace=self.ws, hspace=self.hs)
        a_jct = fig.add_subplot(gr[0, 0])
        a_ppe = fig.add_subplot(gr[0, 1])
        a_tnk = fig.add_subplot(gr[1, :])

        self.plot_junction(junction, a_jct)
        self.plot_pipe(pipe, a_ppe)
        self.plot_tank(tank, a_tnk)

        self.generate_legend(fig)

        if pdf_f is not None:
            pdf_f.savefig()
            plt.close()

    def station_tank_graph(self, tank, site_name=None, pdf_f=None):
        fig = plt.figure()
        fig.set_size_inches(17, 11)
        if site_name is not None:
            fig.suptitle("Site - {}".format(site_name))

        gr = plt.GridSpec(1, 1)
        a_tnk = fig.add_subplot(gr[0, 0])
        self.plot_tank(tank, a_tnk)

        self.generate_legend(fig)

        if pdf_f is not None:
            pdf_f.savefig()
            plt.close()


    def station_junction_only(self, junction, site_name=None, pdf_f=None):
        fig = plt.figure()
        fig.set_size_inches(17, 11)
        if site_name is not None:
            fig.suptitle("Site - {}".format(site_name))

        gr = plt.GridSpec(1, 1)
        a_jct = fig.add_subplot(gr[0, 0])
        self.plot_junction(junction, a_jct)

        self.generate_legend(fig)

        if pdf_f is not None:
            pdf_f.savefig()
            plt.close()

    def station_pipe_only(self, pipe, site_name=None, pdf_f=None):
        fig = plt.figure()
        fig.set_size_inches(17, 11)
        if site_name is not None:
            fig.suptitle("Site - {}".format(site_name))

        gr = plt.GridSpec(1, 1)
        a_ppe = fig.add_subplot(gr[0, 0])
        self.plot_pipe(pipe, a_ppe)

        self.generate_legend(fig)

        if pdf_f is not None:
            pdf_f.savefig()
            plt.close()


    def plot_junction(self, junction, ax):
        ax.set_title("Junction")

        y_field = "HEAD"
        self.plot_generic(junction, ax, 'junctions', y_field)

        # ax.legend()
        ax.grid()
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Head (m)")

        self.test_lims(ax)

    def plot_pipe(self, pipe, ax):
        ax.set_title("Pipe")
        y_field = "FLOW"

        self.plot_generic(pipe, ax, 'pipes', y_field)

        # test if all negative
        self._validate_sign(ax)

        # ax.legend()
        ax.grid()
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Flow (l/s)")

        self.test_lims(ax)

    def plot_tank(self, tank, ax):
        ax.set_title("Tank")

        y_field = "F__VOLUME"
        
        p = "tanks"

        self.plot_generic(tank, ax, p, y_field)

        ax.grid()
        ax.set_xlabel("Time Step")
        ax.set_ylabel("% FULL")

        self.test_lims(ax)
        if ax.get_ylim()[1] > 100:
            ax.set_ylim((ax.get_ylim()[0], 100.2))

    def plot_generic(self, data, ax, p, y_field):
        ls = [
            [self.new_scenario_data, "New_Id", self.cls["NewData"], self.new_model_label],
            [self.old_scenario_data, "Old_Id", self.cls["OldData"], self.old_model_label],
        ]
        for i in ls:
            scenario_data, id_f, color, lbl = i
            s_data = scenario_data[p]
            s_data = s_data.loc[s_data["ID"].isin(data[id_f])].set_index(
                "TIME_STEP")
            ax.plot(s_data[y_field], label=lbl, color=color)
            self.plot_daily_avg(s_data[y_field], ax, color, lbl)

        if self.sc is not None:
            if not data["SCADA_TAG"].isna().any():
                try:
                    scada_tag = data["SCADA_TAG"]
                    scada_data = self.sc
                    scada_data = scada_data.set_index(scada_data.index.hour, append=True)[scada_tag].unstack()
                    scada_data.plot.box(showfliers=False, notch=True, ax=ax, color=self.cls['SCADA'])

                    scada_med =self.sc[scada_tag].groupby(self.sc[scada_tag].index.hour).median()
                    ax.plot(scada_med.index+1,scada_med.values,color=self.cls['SCADA'], label="SCADA")

                    self.plot_daily_avg(self.sc[scada_tag], ax, self.cls['SCADA'], "SCADA")
                    
                except KeyError:
                    print(f"No data for tag {scada_tag.values[0]}")
                except:
                    print("SCADA problem with {}".format(data["New_Id"]))

        ax.set_xlim((1, 24))
        ax.set_xticks(list(range(1, 24)))
        ax.set_xticklabels(list(range(1, 24)))

    def test_lims(self, ax):
        yl = ax.get_ylim()

        if yl[1] - yl[0] < 5:
            mid = (yl[1] - yl[0]) + yl[0]
            b = round(mid - 5, 1)
            t = round(mid + 5, 1)
            ax.set_ylim((b, t))

    def generate_legend(self, fig):
        handles = []
        labels = []
        for a in fig.axes:
            a.legend().remove()
            
            _h, _l = a.get_legend_handles_labels()
            handles += _h
            labels += _l

        by_label = OrderedDict(zip(labels, handles))

        fig.subplots_adjust(top=0.88)

        c_num = int(max(1, len(by_label.keys())/2))

        fig.legend(
            by_label.values(), by_label.keys(), 
            loc='center', bbox_to_anchor=(0.5, 0.94), ncol=c_num,
            fancybox=True, shadow=True
        )

    def _validate_sign(self, ax):
        """
        checks to see if all the tags in the graph are negative, ignoring scada
        if all values in the series are negative, in both the new and old, flips the sign to be positive
        This really should only apply to pipes, specifically pipes that are drawn "reverse" 

        Problem is i get a funny recast error with this and i don't know why
        Might be related to the 2 value limits or something else.

        Either way probably best to just fix it in the model by reversing the pipe to the appropriate direction.
        I have put this as a flag to print out instead, but kept code for posterit 
        """
        vals = []
        for l in ax.get_lines():
            _v = list(l.get_ydata())
            if "SCADA" in l.get_label().upper():
                continue
            vals += _v

        if not ((min(vals) >= 0) or (max(vals) >= 0)):
            arcpy.AddMessage("Data may be reversed. Check model pipe directions to correct sign")

            # for l in ax.get_lines():
            #     if "SCADA" in l.get_label().upper():
            #         continue
            #     l.set_ydata(l.get_ydata()*(-1))

            # all_vals = abs(np.array(all_vals))
            # ax.set_ylim(min(all_vals)-1, max(all_vals)+1)

            # ax.relim()
            # ax.autoscale_view()

    def plot_daily_avg(self, data, ax, color, label):
        nd = data.median()
        if type(nd) is pd.Series:
            nd = nd.median()
        return ax.axhline(linewidth=2, color=color, y=nd, alpha=0.5, linestyle='dashed',
                   label="{} - Median".format(label))

    def zone_base(self, zone_data, pdf_f=None):
        poi_data = zone_data.loc[(zone_data["Site"] == "POI")]
        data = zone_data.loc[~(zone_data["Site"] == "POI")]
        _sites = sorted(data["Site"].unique())
        for i_enum, site in enumerate(_sites):
            arcpy.AddMessage(f"Site {i_enum+1} of {len(_sites)}\t{site}")
            self.site_base(data.loc[data["Site"] == site], pdf_f)

        arcpy.AddMessage("Plotting Points of Interest for zone")
        self.POI_chart(poi_data, pdf_f)

    def site_base(self, site_data, pdf_f=None):
        site_name = list(site_data["Site"])[0]
        site_junction = site_data.loc[site_data["Type"] == "Junction"]
        site_pipe = site_data.loc[site_data["Type"] == "Pipe"]
        site_tank = site_data.loc[site_data["Type"] == "Tank"]

        # arcpy.AddMessage(f"Site: {site_name}")

        # check for a tank
        if site_tank.count().max() > 0:
            if any([site_junction.count().max(), site_pipe.count().max()]) > 0:
                self.station_with_tank_graph(site_junction, site_pipe, site_tank,
                                        site_name, pdf_f=pdf_f)
            else:
                self.station_tank_graph(site_tank, site_name, pdf_f=pdf_f)
        else:
            # no tank, plot without.
            if (site_junction.count().max() > 0) and (site_pipe.count().max() == 0):
                self.station_junction_only(site_junction, site_name, pdf_f=pdf_f)
            elif (site_junction.count().max() == 0) and (site_pipe.count().max() > 0):
                self.station_pipe_only(site_pipe, site_name, pdf_f=pdf_f)
            else:
                self.station_without_tank_graph(site_junction, site_pipe, site_name,
                                           pdf_f=pdf_f)
    # endregion
