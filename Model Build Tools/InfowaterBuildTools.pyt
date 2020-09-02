# -*- coding: utf-8 -*-

import logging
import os
import arcpy
import sys
import copy

import pandas as pd
import numpy as np
import random

arcpy.env.overwriteOutput = True

py3 = False
if sys.version_info[0] == 3:
    py3 = True


def arc_log_output(message, m_type=None):
    """
    this is to fix the logging calls and the arcpy message calls
    if not m_type provided, defaults to INFO.
    message: type(string)
    m_type: one of INFO, WARNING, ERROR for m_type
    """
    if m_type is None:
        m_type = "INFO"

    if m_type == "INFO":
        logging.info(message)
        arcpy.AddMessage(message)
    elif m_type == "WARNING":
        logging.warning(message)
        arcpy.AddWarning(message)
    elif m_type == "ERROR":
        logging.error(message)
        arcpy.AddError(message)


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "InfowaterBuildTools"
        self.alias = "Infowater Build Tools"

        # List of tool classes associated with this toolbox
        self.tools = [
            SplitWatermains,
            DemandIsolationNodes,
            ClosedPipes,
            JunctionElevations,
        ]


class SplitWatermains(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Split Watermains"
        self.description = "Detect and split watermains at locations. "
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
            displayName="Line Feature",
            name="line_feature",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )
        param0.filter.list = ["Polyline"]

        param1 = arcpy.Parameter(
            displayName="Output Geodatabase",
            name="output_gdb",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input",
        )
        param1.filter.list = ["Local Database"]

        param2 = arcpy.Parameter(
            displayName="Snap Distance",
            name="snap_distance",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input",
        )

        param3 = arcpy.Parameter(
            displayName="Temporary Output",
            name="temp_output",
            datatype="DEWorkspace",
            parameterType="Optional",
            direction="Input",
        )

        param4 = arcpy.Parameter(
            displayName="Generate Split Locations Only",
            name="split_points_only",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )

        param5 = arcpy.Parameter(
            displayName="Output Feature",
            name="output_feature",
            datatype="GPFeatureLayer",
            parameterType="Derived",
            direction="Output",
        )

        # param0.value = r'I:\Cody\Mapping\SegmentationTool\toolbox\tool_input.gdb\water_mains'
        # param1.value = r'I:\Cody\Mapping\SegmentationTool\toolbox\tool_output.gdb'
        # if py3:
        #     param0.value = param0.value.replace("I:", "J:")
        #     param1.value = param1.value.replace("I:", "J:")

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
        line_feature = parameters[0].valueAsText
        output_location = parameters[1].valueAsText

        snap_distance = None
        if parameters[2].value is not None:
            snap_distance = float(parameters[2].valueAsText)

        temp_output = None
        if parameters[3].value is not None:
            temp_output = parameters[3].valueAsText

        split_points_only = None
        if parameters[4].value is not None:
            if parameters[4].valueAsText == "true":
                split_points_only = True

        output_feature = None

        # pn = [p.name for p in parameters]
        # pv = [line_feature, output_location, snap_distance, temp_output, split_points_only, output_feature]
        # pv = ["{}".format(p) for p in pv]
        # msg = "\n----\n".join(["\n".join(p) for p in zip(pn, pv)])
        # arc_log_output(msg)

        self._do_work(
            line_feature,
            output_location,
            snap_distance,
            temp_output,
            test_run=split_points_only
        )

        return

    def _do_work(self, feature, output_location, distance=None, _temp_location=None, **kwargs):
        arcpy.env.overwriteOutput = True
        if _temp_location is None:
            if py3:
                _temp_location = "memory"
            else:
                _temp_location = "in_memory"

        arc_log_output("Temporary Output Location: {}".format(_temp_location))

        if distance is None:
            distance = "0.01 METERS"
        arc_log_output("Distance set to {}".format(distance))

        feature = self.copy_features(feature, _temp_location)
        feature_lyr = self.create_feature_layer(feature, "F1")

        self.snap_features(feature_lyr, distance)
        # self.save_output(feature_lyr, "snap_tmp", _temp_location)
        # feature_lyr = self.create_feature_layer(feature, "F2")

        split_locations = self.calculate_split_locations(
            feature_lyr, output_location, distance, _temp_location
        )

        if kwargs.get("test_run") is not None:
            arc_log_output("Running in test mode. Returning")
            return

        self.split_features(feature_lyr, split_locations)
        self.save_output(feature, "NewSplitFeature", output_location)

    def save_output(self, feature, name, location):
        arcpy.CopyFeatures_management(feature, os.path.join(location, name))

    def create_feature_layer(self, feature, fl_name=None):
        if fl_name is None:
            fl_name = "{}_FL".format(feature.split("\\")[-1])
        arc_log_output("Feature Layer Name: {}".format(fl_name))
        return arcpy.MakeFeatureLayer_management(feature, fl_name)[0]

    def calculate_split_locations(self, feature, output_location, distance, _temp_location):
        arc_log_output("Calculating Split Locations")
        verticies = self.get_verticies(feature)
        vertex_feature = self.vertices_to_feature(verticies, _temp_location)
        vertex_events = self.collect_points(vertex_feature, _temp_location)
        vertex_candidates = self.get_candidates(
            vertex_events, feature, _temp_location, distance, output_location
        )
        return vertex_candidates

    def copy_features(self, feature, output_location):
        in_feat = feature
        nm = in_feat.split("\\")[-1]
        out_feat = os.path.join(output_location, nm)
        arc_log_output("Copying input feature to {}".format(out_feat))
        if in_feat == out_feat:
            raise Exception("Can't have same location")
        return arcpy.CopyFeatures_management(in_feat, out_feat)[0]

    def snap_features(self, feature, distance):
        """
        snap the feature to the edge to correct any issues.
        This is similar to the process to find the locations where there is connectivity issues.
        should do end snap, but i think i could clean that up previously by snapping end to end
        """
        arc_log_output(
            "Snapping Feature {} - Distance {}".format(feature, distance))
        snap_env = [
            [feature, "END", distance],
            [feature, "VERTEX", distance],
            [feature, "EDGE", distance],
        ]
        arcpy.Snap_edit(feature, snap_env)[0]

    def get_verticies(self, feature):
        arc_log_output("Getting Verticies")
        out_feats = []
        errors = []
        sr = arcpy.SpatialReference(26917)
        with arcpy.da.SearchCursor(feature, ["OID@", "SHAPE@"]) as _sc:
            for row in _sc:
                try:
                    cent = row[1]
                    fp = arcpy.PointGeometry(cent.firstPoint, sr)
                    lp = arcpy.PointGeometry(cent.lastPoint, sr)

                    out_feats.append([row[0], fp])
                    out_feats.append([row[0], lp])
                except:
                    arc_log_output(
                        "Problem with feature {}".format(row[0]), "ERROR")
                    errors.append(row[0])

        return out_feats

    def vertices_to_feature(self, vts, temp_folder):
        """Generate a feature class of supplied vertices"""
        arc_log_output("Converting vertices to feature")
        ft = arcpy.CreateFeatureclass_management(
            temp_folder, "verts", "POINT", spatial_reference=arcpy.SpatialReference(26917))[0]
        arcpy.AddField_management(ft, "ID", "TEXT")
        with arcpy.da.InsertCursor(ft, ["ID", "SHAPE@"]) as ic:
            for item in vts:
                ic.insertRow((item[0], item[1],))
        return ft

    def collect_points(self, verts, temp_folder):
        """Get locations where only one vertices is present"""
        arc_log_output("Collecting Points")
        ev = arcpy.CollectEvents_stats(
            verts,
            os.path.join(temp_folder, "events_1"))[0]

        s_ev = arcpy.Select_analysis(
            ev,
            os.path.join(temp_folder, "cand_events"), "\"ICOUNT\" = 1")[0]
        return s_ev

    def get_candidates(self, verts, pipes, temp_folder, join_distance, out_folder):
        arc_log_output("Getting Locations to Check")
        sj = arcpy.SpatialJoin_analysis(verts, pipes,
                                        os.path.join(
                                            temp_folder, "join_events"),
                                        "JOIN_ONE_TO_ONE", "KEEP_ALL",
                                        "", "INTERSECT", "0 METERS")[0]
        eee = arcpy.Select_analysis(sj,
                                    os.path.join(
                                        out_folder, "SplitPoints"),
                                    "\"JOIN_COUNT\" = 2")[0]
        return eee

    def _get_feature_fields(self, feature):
        arc_log_output("getting fields")
        ignore_field_names = ['SHAPE_LENGTH']
        ignore_field_types = ["OID", "GEOMETRY"]
        flds = [fld.name for fld in arcpy.ListFields(feature)
                if fld.name.upper() not in ignore_field_names and fld.type.upper() not in ignore_field_types]
        return flds

    def split_features(self, line_features, point_features):
        # get field list (removing OID + shape obj)
        # should i be taking out a transactional edit here before making changes?
        # doesn't really matter since working on memory workspace (in theory)
        arc_log_output("splitting features")
        fields = self._get_feature_fields(line_features)

        total = int(arcpy.GetCount_management(point_features)[0])
        ip = int(total*0.1)
        if ip < 20:
            ip = 1

        arcpy.SetProgressor('step', "Splitting Features", 0, total, 1)

        with arcpy.da.SearchCursor(point_features, ["SHAPE@", "OID@"]) as sc:
            for ix, row in enumerate(sc):
                ix = ix+1
                # arc_log_output("{} of {}".format(ix, total))
                if (ix % ip) == 0:
                    arc_log_output("{} of {}".format(ix, total))
                    arcpy.SetProgressorPosition(ix)

                _shp = row[0]
                # select by location - points against lines
                arcpy.SelectLayerByLocation_management(
                    line_features, "INTERSECT", _shp, "0 METERS", "NEW_SELECTION")

                if int(arcpy.GetCount_management(line_features)[0]) == 0:
                    arc_log_output("No feature selected {}".format(row[1]))
                    continue

                # do split operation
                res = self.split_line_at_point(line_features, _shp, fields)

                if res is False:
                    arc_log_output(
                        "OID {} not split".format(row[1]), "WARNING")

        arcpy.SetProgressorPosition(total)
        arcpy.ResetProgressor()
        return line_features

    def split_line_at_point(self, line_feature, point_feature, fields):
        # the input here will likely be a pre-selected limited feature class
        # should return 2 lines selected. 1 for end point feature, 1 for cut feature
        # check the percentage along line != 1 and cut that line
        full_field_list = ["SHAPE@"] + fields
        _shp, _field_data = None, None
        new_a, new_b = None, None
        with arcpy.da.UpdateCursor(line_feature, full_field_list) as sc:
            for row in sc:
                _shp = row[0]
                _field_data = row[1:]
                a = copy.copy(_shp)
                b = copy.copy(_shp)

                percentage_along_line = a.measureOnLine(point_feature, True)
                if (percentage_along_line == 1) or (percentage_along_line == 0):
                    continue

                new_a = a.segmentAlongLine(0, percentage_along_line, True)
                new_b = b.segmentAlongLine(percentage_along_line, 1, True)

                sc.updateRow([new_a] + _field_data)

        if new_a is None or new_b is None:
            return False

        with arcpy.da.InsertCursor(line_feature, full_field_list) as ic:
            ic.insertRow([new_b] + _field_data)

        return True


class DemandIsolationNodes(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Demand and Isolation Nodes"
        self.description = "tool to flag which nodes should be demand or isolation"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
            displayName="Node Feature",
            name="node",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )
        param0.filter.list = ["Point"]

        param1 = arcpy.Parameter(
            displayName="Line Feature",
            name="line",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )
        param1.filter.list = ["Polyline"]

        param2 = arcpy.Parameter(
            displayName="Output Geodatabase",
            name="output_gdb",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input"
        )

        param3 = arcpy.Parameter(
            displayName="Closed Pipe List",
            name="ppe_list",
            datatype=["DEFeatureClass", "DEFile"],
            # datatype="DEFile",
            parameterType="Optional",
            direction="Input"
        )
        # param3.filter.list = ["Polyline", 'txt', 'csv']

        param4 = arcpy.Parameter(
            displayName="Search Distance",
            name="search_distance",
            datatype="GPLinearUnit",
            parameterType="Optional",
            direction="Input"
        )
        param4.value = "50 Meters"

        param5 = arcpy.Parameter(
            displayName="Output Feature",
            name="output_feature",
            datatype="GPFeatureLayer",
            parameterType="Derived",
            direction="Output",
        )

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
        if py3:
            self.mem_path = "memory"
        else:
            self.mem_path = "in_memory"

        node = parameters[0].valueAsText
        pipe = parameters[1].valueAsText
        output_gdb = parameters[2].valueAsText
        ppe_list = parameters[3].valueAsText
        distance = parameters[4].valueAsText

        # self.mem_path = output_gdb

        result = self.do_work(node, pipe, output_gdb, ppe_list, distance)
        arcpy.SetParameter(5, result)
        return

    def do_work(self, node_feature, pipe_feature, output_gdb, pipe_closed_list=None, distance=None):
        arcpy.env.overwriteOutput = True

        if distance is None:
            distance = "50 Meters"

        arc_log_output("Building ID List")
        id_df = self.build_id_list(node_feature, "ID")

        arc_log_output("Creating Feature Layers")
        node_fl = arcpy.MakeFeatureLayer_management(node_feature, "NODE_FL")[0]
        ppe_fl = arcpy.MakeFeatureLayer_management(pipe_feature, "PPE_FL")[0]

        # calculate first batch
        arc_log_output("Calculating First Batch")
        dmd_id = self.calculate_first_stage(node_fl, ppe_fl)
        id_df.loc[id_df["ID"].isin(dmd_id), "Flag"] = "Demand"

        # might have to do this after first stage
        # causes tees to become
        if pipe_closed_list is not None:
            # do some stuff to create a new feature layer with pipes closed
            # generate list of closed pipes
            # create pipe feature that doesn't contain closed pipes
            # this should make any closed pipe a dead end that will flag demandy
            arc_log_output("Calculating Pipe Closure Batch")
            ppe_id_list = self.load_closed_pipes(pipe_closed_list)

            ppe_sql = "\"ID\" NOT IN ({})".format(
                ",".join(["'{}'".format(x) for x in ppe_id_list]))
            ppe_fl = arcpy.MakeFeatureLayer_management(
                pipe_feature, "PPE_FL", ppe_sql)
            dmd_id = self.calculate_first_stage(node_fl, ppe_fl)
            id_df.loc[id_df["ID"].isin(dmd_id), "Flag"] = "Demand"

        # calculate second batch
        arc_log_output("Calculating Junction to Junction batch")
        second_id_list = list(id_df.loc[id_df["Flag"].isnull()]["ID"].values)
        sql = "\"ID\" IN ({})".format(
            ", ".join(["'{}'".format(x) for x in second_id_list]))
        node_fl_2 = arcpy.MakeFeatureLayer_management(
            node_fl, "NODE_FL_2", sql)

        dmd_id = self.calculate_second_stage(
            node_fl_2, node_fl, id_df, distance)
        id_df.loc[id_df["ID"].isin(dmd_id), "Flag"] = "Demand"
        # arcpy.AddMessage(id_df.loc[id_df["Flag"].isnull()])
        id_df.loc[id_df["Flag"].isnull(), "Flag"] = "Isolation"

        arc_log_output("Saving output")
        fc = self.save_output(node_fl, id_df, output_gdb)

        return fc

    def load_closed_pipes(self, pipe_closed_list):
        ppe_id_list = None

        fl_types = {'.txt': '\t', '.csv': ','}
        if any([x in pipe_closed_list for x in fl_types]):
            sep = None
            for i in fl_types:
                if i in pipe_closed_list:
                    sep = fl_types.get(i)
            df = pd.read_csv(pipe_closed_list, sep=sep)

            ppe_df = pd.read_csv(pipe_closed_list)
            ppe_id_list = list(ppe_df["ID"].values)

        else:
            with arcpy.da.SearchCursor(pipe_closed_list, ["ID"]) as sc:
                ppe_id_list = [row[0] for row in sc]

        return ppe_id_list

    def calculate_first_stage(self, nodes, pipes):
        join = arcpy.SpatialJoin_analysis(
            nodes, pipes, "{}\\join_1".format(self.mem_path),
            "JOIN_ONE_TO_MANY", "KEEP_ALL",
        )[0]

        with arcpy.da.SearchCursor(join, "ID") as sc:
            id_list = [row[0] for row in sc]

        id_df = pd.DataFrame(id_list, columns=["ID"])
        id_df = id_df["ID"].value_counts()

        id_list = id_df.loc[id_df != 2].index

        return id_list

    def calculate_second_stage(self, check_nodes, all_nodes, id_dataframe, distance):
        join = arcpy.SpatialJoin_analysis(
            check_nodes, all_nodes, "{}\\join_2".format(self.mem_path),
            "JOIN_ONE_TO_MANY", "KEEP_ALL",
            search_radius=distance
        )[0]

        with arcpy.da.SearchCursor(join, ["ID", "ID_1"]) as sc:
            id_list = [row for row in sc]

        result_df = pd.DataFrame(id_list, columns=["ID", "ID_1"])
        result_vc = result_df["ID"].value_counts()

        demand = list(result_df.loc[result_df["ID"].isin(
            result_vc.loc[result_vc == 1].index)]["ID"].values)
        isolation = list(result_df.loc[result_df["ID"].isin(
            result_vc.loc[result_vc > 2].index)]["ID"].values)

        two_df = result_df.loc[result_df["ID"].isin(
            result_vc.loc[result_vc == 2].index)][["ID", "ID_1"]]

        for i in two_df.loc[~(two_df["ID"] == two_df["ID_1"])].iterrows():
            series = i[1]
            val_list = []
            id_1 = series["ID"]
            id_2 = series["ID_1"]

            val_list = []
            # this is a flag to see if its in my previously checked and flagged values (from previous steps)
            main_flag = False
            for v in [id_1, id_2]:
                # check if the id is in previous step review; change main flag value to True if so
                if v in id_dataframe.loc[~id_dataframe["Flag"].isnull()]["ID"].values:
                    main_flag = True
                    continue

                # check if i have already assigned this node to either demand or isolation previously; ignore if so
                if (v in demand) or (v in isolation):
                    continue

                # both previous checks passed; add to value list
                val_list.append(v)

            # check if list is empty aka all values are previously assigned; continue to next pair
            if len(val_list) == 0:
                continue

            # check for main flag, set all nodes isolation if true; else select sample from pair for demand and set all others (should be 1) to isolation
            if main_flag:
                dmd = []
                iso = val_list
            else:
                dmd = random.sample(val_list, 1)
                iso = [v for v in val_list if v not in dmd]

            demand += dmd
            isolation += iso
        return demand

    def save_output(self, nodes, id_dataframe, output_gdb):
        # arcpy.AddMessage(id_dataframe)
        id_dictionary = id_dataframe.set_index("ID").to_dict()
        id_dictionary = id_dictionary.get("Flag", None)
        # arcpy.AddMessage(id_dictionary)

        fc = arcpy.CreateFeatureclass_management(output_gdb, "Demand_Isolation_Nodes", "POINT",
                                                 spatial_reference=arcpy.SpatialReference(
                                                     26917)
                                                 )
        arcpy.AddField_management(fc, "ID", "TEXT")
        arcpy.AddField_management(fc, "Flag", "TEXT")

        with arcpy.da.InsertCursor(fc, ["SHAPE@", "ID", "Flag"]) as ic:
            with arcpy.da.SearchCursor(nodes, ["SHAPE@", "ID"]) as sc:
                for row in sc:
                    _shp = row[0]
                    _id = row[1]
                    _flag = id_dictionary.get(_id, None)
                    ic.insertRow((_shp, _id, _flag,))

        return fc

    def build_id_list(self, feature, field):
        with arcpy.da.SearchCursor(feature, field) as sc:
            id_list = [row[0] for row in sc]
        return pd.DataFrame(id_list, columns=["ID"])


class ClosedPipes(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Calculate Closed Pipes"
        self.description = "calculate the location of closed pipes in network"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
            displayName="Pipe Feature",
            name="pipe_feature",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )
        param0.filter.list = ["Polyline"]

        param1 = arcpy.Parameter(
            displayName="Pressure Zone Feature",
            name="pzone_feature",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )

        param2 = arcpy.Parameter(
            displayName="Output Geodatabase",
            name="output_gdb",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input"
        )

        param3 = arcpy.Parameter(
            displayName="Pipe ID Field",
            name="fields",
            datatype="Field",
            parameterType="Required",
            direction="Input",
            multiValue=False
        )
        param3.parameterDependencies = [param0.name]

        param4 = arcpy.Parameter(
            displayName="Valve Feature",
            name="valve_feature",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input"
        )
        param5 = arcpy.Parameter(
            displayName="Valve Closure Field",
            name="closure_field",
            datatype="Field",
            parameterType="Optional",
            direction="Input"
        )
        param5.parameterDependencies = [param4.name]
        param5.value = "OperationalStatus"

        param_out = arcpy.Parameter(
            displayName="test out",
            name="testout",
            datatype="GPFeatureLayer",
            parameterType="Derived",
            direction="Output",
        )

        params = [
            param0, param3, param1, param2,
            param4, param5,
            param_out
        ]
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
        if py3:
            self.mem_path = "memory"
        else:
            self.mem_path = "in_memory"

        arcpy.env.overwriteOutput = True

        ppe_feature = parameters[0].valueAsText
        self.id_field = parameters[1].valueAsText

        pressure_zone_feature = parameters[2].valueAsText
        output_gdb = parameters[3].valueAsText


        valve_feature = parameters[4].valueAsText
        valve_field = parameters[5].valueAsText

        self.mem_path = output_gdb

        self.do_work(ppe_feature, pressure_zone_feature,
                     output_gdb, valve_feature, valve_field)

        return

    def do_work(self, pipe_feature, pressure_zone_feature, output_gdb, valve_feature=None, valve_field=None):
        # set index for whole class
        self.watermain_index = self.index_watermains(
            pipe_feature, self.id_field)
        pipe_feature_lyr = arcpy.MakeFeatureLayer_management(pipe_feature, "PPE_FL")[0]

        # calculate valve closures
        closed_vlv = []
        if valve_feature is not None:
            vlv_sql = "\"{}\" LIKE 'Closed'".format(valve_field)
            vlv_fc = arcpy.MakeFeatureLayer_management(
                valve_feature, "VLV_FC", vlv_sql)
            closed_vlv, filter_wm = self.closed_valves(pipe_feature_lyr, vlv_fc)

            # this removes the pipes found in the closed valve loop from the pressure zone closure step
            sql = "\"{}\" NOT IN ({})".format(
                self.id_field, ", ".join(["'{}'".format(x) for x in filter_wm]))
            pipe_feature_lyr = arcpy.MakeFeatureLayer_management(pipe_feature, "PPE_FL", sql)

            # can remove
            ff = arcpy.CopyFeatures_management(
                pipe_feature_lyr, os.path.join(self.mem_path, "PVLV"))


        cpipe = self.calculate_closed_pipes(
            pipe_feature_lyr, pressure_zone_feature)

        # add closed pipe dataset; could be nothing if valves weren't provided 
        cpipe += closed_vlv
        
        sql = "\"{}\" IN ({})".format(
            self.id_field, ", ".join(["'{}'".format(x) for x in cpipe]))
        closed_pipe_feature = arcpy.MakeFeatureLayer_management(
            pipe_feature, "POUT_FL", sql)[0]

        # can remove
        ff = arcpy.CopyFeatures_management(
            closed_pipe_feature, os.path.join(self.mem_path, "PCPIPE"))
        # 

        # calculate dead ends
        pipe_feature_lyr = arcpy.MakeFeatureLayer_management(pipe_feature, "PPE_FL")[0]
        de = self.remove_dead_ends(pipe_feature_lyr, closed_pipe_feature)

        # re build pipes with dead ends removed
        cpipe = [c for c in cpipe if c not in de]

        sql = "\"{}\" IN ({})".format(
            self.id_field, ", ".join(["'{}'".format(x) for x in cpipe]))
        closed_pipe_feature = arcpy.MakeFeatureLayer_management(
            pipe_feature, "POUT_FL", sql)[0]

        ff = arcpy.CopyFeatures_management(
            closed_pipe_feature, os.path.join(output_gdb, "ClosedPipes"))

        arcpy.SetParameter(4, ff)

    def index_watermains(self, feature, id_field):
        with arcpy.da.SearchCursor(feature, [id_field, "SHAPE@LENGTH"]) as sc:
            return {row[0]: row[1] for row in sc}

    def closed_valves(self, pipe_feature, valve_feature):
        sj = arcpy.SpatialJoin_analysis(
            valve_feature, pipe_feature, "{}\\vlv_join".format(self.mem_path),
            "JOIN_ONE_TO_MANY", "KEEP_COMMON",
            search_radius="0.1 Meters"
        )

        sj_index = self.build_spatial_join_index(sj)

        close_ppe = []
        for i in sj_index:
            f = self.determine_closed_feature(sj_index[i])
            close_ppe.append(f)

        all_join_ppe = [v for r in sj_index for v in sj_index[r]]
        return close_ppe, all_join_ppe

    # region closed pipes

    def calculate_closed_pipes(self, pipe_feature_layer, pressure_zone_feature):
        arc_log_output("Calculating Closed Boundary Pipes")

        arc_log_output("Building Edge Feature")
        pzone_edge = self.create_pressure_zone_boundary(pressure_zone_feature)

        arcpy.SelectLayerByLocation_management(
            pipe_feature_layer, "INTERSECT", pzone_edge, "0.1 Meters", "NEW_SELECTION")

        ppe_fl = arcpy.MakeFeatureLayer_management(
            pipe_feature_layer, "ppe_boundary")[0]

        arc_log_output("Getting Boundary Pipes")
        ppe_diss = arcpy.Dissolve_management(
            ppe_fl, "{}\\ppe_diss".format(self.mem_path),
            None, None, "MULTI_PART", "UNSPLIT_LINES"
        )[0]

        sj = arcpy.SpatialJoin_analysis(
            ppe_diss, ppe_fl, "{}\\sj".format(self.mem_path),
            "JOIN_ONE_TO_MANY", "KEEP_COMMON",
            # search_radius="0.1 meters",
        )

        arc_log_output("Determining closures")
        sj_index = self.build_spatial_join_index(sj)

        close_ppe = []
        for i in sj_index:
            f = self.determine_closed_feature(sj_index[i])
            close_ppe.append(f)
        return close_ppe

    def build_spatial_join_index(self, spatial_join_feature):
        dct = {}
        with arcpy.da.SearchCursor(spatial_join_feature, ["TARGET_FID", self.id_field]) as sc:
            for row in sc:
                v = dct.get(row[0], [])
                v.append(row[1])
                dct[row[0]] = v
        return dct

    def determine_closed_feature(self, sj_index, wm_index=None):
        if wm_index is None:
            wm_index = self.watermain_index
        min_ppe = None
        min_ppe_len = None
        for val in sj_index:
            p_len = wm_index.get(val)
            if min_ppe is None:
                min_ppe = val
                min_ppe_len = p_len
                continue

            if p_len < min_ppe_len:
                min_ppe = val
                min_ppe_len = p_len

        return min_ppe

    def create_pressure_zone_boundary(self, pressure_zone_feature):
        """
        Creates a unionized polygon of the pressure zone boundaries as one single item
        :return:
        :rtype:
        """
        lines_array = self.get_pressure_zone_lines(pressure_zone_feature)
        polyline_union = self.union_pressure_zones(lines_array)
        return polyline_union

    def union_pressure_zones(self, lines_array):
        """
        Unionize all the individual line segments within the pressure zone lines array
        :param lines_array:
        :type lines_array:
        :return:
        :rtype:
        """
        polyline_union = None
        for l in lines_array:
            if polyline_union is None:
                polyline_union = l
                continue
            polyline_union = polyline_union.union(l)
        return polyline_union

    def get_pressure_zone_lines(self, pressure_zone):
        """
        Get each individual line segment of the pressure zones
        :return:
        :rtype:
        """
        sr = arcpy.SpatialReference(26917)

        lines_array = []
        pressure_zone = arcpy.Dissolve_management(
            pressure_zone, r'in_memory\PzoneDissolve', ["Zone"])
        with arcpy.da.SearchCursor(pressure_zone, ["SHAPE@", "ZONE"]) as sc:
            for row in sc:
                for part in row[0]:
                    lines_array.append(arcpy.Polyline(part, sr))
        return lines_array
    # end region

    # region clean dead end closures
    def remove_dead_ends(self, pipe_feature, closed_pipes_feature):
        # only build vertex list of features that are flagged closed.
        # faster and makes sense to only look at options that are going to be modified
        vt_feat = self.vertices_to_feature(
            self.get_verticies(closed_pipes_feature))
        sj_feat = arcpy.SpatialJoin_analysis(
            vt_feat, pipe_feature, "{}\\cpsj".format(self.mem_path),
            "JOIN_ONE_TO_MANY", "KEEP_COMMON"
        )

        with arcpy.da.SearchCursor(sj_feat, ["TARGET_FID", self.id_field]) as sc:
            data = [r for r in sc]
        data = pd.DataFrame(data, columns=["FID", "ID"])
        df_vc = data["FID"].value_counts()
        df_vc = df_vc.loc[df_vc == 1]
        data = list(data.loc[data["FID"].isin(df_vc.index)]["ID"].values)

        return data

    def get_verticies(self, feature):
        arc_log_output("Getting Verticies")
        out_feats = []
        errors = []
        sr = arcpy.SpatialReference(26917)
        with arcpy.da.SearchCursor(feature, [self.id_field, "SHAPE@"]) as _sc:
            for row in _sc:
                try:
                    cent = row[1]
                    fp = arcpy.PointGeometry(cent.firstPoint, sr)
                    lp = arcpy.PointGeometry(cent.lastPoint, sr)

                    out_feats.append([row[0], fp])
                    out_feats.append([row[0], lp])
                except:
                    arc_log_output(
                        "Problem with feature {}".format(row[0]), "ERROR")
                    errors.append(row[0])
        return out_feats

    def vertices_to_feature(self, vts):
        """Generate a feature class of supplied vertices"""
        arc_log_output("Converting vertices to feature")
        ft = arcpy.CreateFeatureclass_management(
            self.mem_path, "verts", "POINT", spatial_reference=arcpy.SpatialReference(26917))[0]
        arcpy.AddField_management(ft, self.id_field, "TEXT")
        with arcpy.da.InsertCursor(ft, [self.id_field, "SHAPE@"]) as ic:
            for item in vts:
                ic.insertRow((item[0], item[1],))
        return ft
        # end region


class JunctionElevations(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Junction Elevations"
        self.description = "calculate elevations of junctions"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""

        param0 = arcpy.Parameter(
            displayName="Node Feature",
            name="nodes",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )
        param0.filter.list = ["Point"]

        param1 = arcpy.Parameter(
            displayName="Contour Feature",
            name="contour",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )
        param1.filter.list = ["Polyline"]

        param2 = arcpy.Parameter(
            displayName="Output GDB",
            name="output_gdb",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input"
        )
        param2.filter.list = ["Local Database"]

        param3 = arcpy.Parameter(
            displayName="Node ID Field",
            name="node_field",
            datatype="Field",
            parameterType="Required",
            direction="Input"
        )
        param3.parameterDependencies = [param0.name]

        param_out = arcpy.Parameter(
            displayName="Output Feature",
            name="output_feature",
            datatype="GPFeatureLayer",
            parameterType="Derived",
            direction="Output",
        )

        params = [
            param0, param1, param2, param3,
            param_out
        ]
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
        self.mem_path = "in_memory"

        junction_feature = parameters[0].valueAsText
        contour_feature = parameters[1].valueAsText
        output_gdb = parameters[2].valueAsText

        junction_id_field = parameters[3].valueAsText

        junction_feature = arcpy.CopyFeatures_management(junction_feature, os.path.join(self.mem_path, "JCT"))
        contour_feature = arcpy.CopyFeatures_management(contour_feature, os.path.join(self.mem_path, "CNT"))

        sj = arcpy.SpatialJoin_analysis(
            junction_feature, contour_feature, "{}\\SJ".format(self.mem_path),
            "JOIN_ONE_TO_ONE", "KEEP_ALL", None, "CLOSEST" 
        )

        ft = arcpy.CreateFeatureclass_management(
            output_gdb, "JunctionElevations", "POINT", 
            spatial_reference=arcpy.SpatialReference(26917)
        )
        arcpy.AddField_management(ft, "ID", "TEXT")
        arcpy.AddField_management(ft, "Elevation", "DOUBLE")
        
        with arcpy.da.InsertCursor(ft, ["ID", "Elevation", "SHAPE@"]) as ic:
            with arcpy.da.SearchCursor(sj, [junction_id_field, "Contour", "SHAPE@"]) as sc:
                [ic.insertRow(row) for row in sc]
        
        arcpy.SetParameter(4, ft)
        return
