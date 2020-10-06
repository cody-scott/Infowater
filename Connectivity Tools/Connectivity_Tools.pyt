import arcpy

import os
import logging
import random

arcpy.env.overwriteOutput = True

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')


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
    elif m_type=="WARNING":
        logging.warning(message)
        arcpy.AddWarning(message)
    elif m_type=="ERROR":
        logging.error(message)
        arcpy.AddError(message)


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "InfowaterConnectivity"
        self.alias = "InfowaterConnectivity"

        # List of tool classes associated with this toolbox
        self.tools = [NodeCheck, RandomSampleImages]
        


class RandomSampleImages(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Random Sample Images"
        self.description = "Generate a random set of images at a specified scale for the input feature. Default will be either rounded down MIN(25% of Count, 100)"
        self.canRunInBackground = False

        # dont like using this but it makes sense.
        # better then passing around variable
        self.all_features = None
        self.scale = None
        self.sample_size = None

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
            displayName="Feature",
            name="input_feature",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )
        # param0.filter.list = ["Point"]

        param1 = arcpy.Parameter(
            displayName="Output Folder",
            name="output_folder",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input",
        )
        param1.filter.list = ["Folder"]

        param2 = arcpy.Parameter(
            displayName="Sample Size",
            name="sample_size",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
        )

        param3 = arcpy.Parameter(
            displayName="Map Scale",
            name="map_scale",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input",
        )
        param3.value = 0.9

        param4 = arcpy.Parameter(
            displayName="Map MXD",
            name="map_mxd",
            datatype="DEMapDocument",
            parameterType="Required",
            direction="Input"
        )

     
        return [
            param0, param1, param2, param3, param4
        ]

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
        ##        arcpy.AddMessage("----")
        ##        [arcpy.AddMessage(x.valueAsText) for x in parameters]
        ##        arcpy.AddMessage("----")
        """The source code of the tool."""
        feature = parameters[0].valueAsText
        output_folder = parameters[1].valueAsText

        mxd = parameters[4].valueAsText

        # calculate sample size
        self.validate_sample_size(parameters[2].valueAsText, feature)

        #validate map scale value and fix to 1.1 or greater
        self.validate_map_scale(parameters[3].valueAsText)

        # determine sample features
        random_features = self.calculate_samples(feature)

        # swap to mxd class
        mxd = arcpy.mapping.MapDocument(mxd)

        self.generate_output(random_features, mxd, output_folder)
        
        return

    def validate_map_scale(self, _scale):
        """check scale to avoid runtime error"""

        """
        Note about scale. scale is bugged.
        if value 1 <= x < 2 scale = 1:0.x
        example 1.1 = 1:0.1 in arcmap

        if value x <= 2 scale = 1:x
        example 2.2 = 1:2.2

        does not look like you can set the scale to be a decimal with leading 1
        """
        
        if _scale is None:
            # self.scale = 0.9
            self.scale = None
            return self.scale

        try:
            _scale = float(_scale)
        except:
            #this type check shouldnt hit from the tool as arc should coerce but i guess i should check it...
            arc_log_output("Input Scale {} is not a valid number".format(_scale), "ERROR")
            raise arcpy.ExecuteError
    
        # adjust the scale if between 1 and 2 to be 0.X
        # hopefully will keep scale correct if bug fixed in 10.5+
        if (_scale >= 1) and (_scale < 2):
            _scale -= 1

        
        # if _scale < 0.9:
        #     msg = "Input Scale {} must be greater then 0.9\nUsing scale 0.9".format(_scale)
        #     arc_log_output(msg, "WARNING")
        # self.scale = max(_scale, 0.9)
        self.scale = _scale
        return self.scale


    def validate_sample_size(self, _input_ss, _feat):
        """calculate sample size to either 25% of total or input size"""
        feat_count = int(arcpy.GetCount_management(_feat)[0])
        _default_ratio = 0.25
        _sample_count = int(feat_count * _default_ratio)


        if _input_ss is not None:
            try:
                _input_ss = int(_input_ss)
            except:
                arc_log_output("Input Sample Size {} is not a number".format(_input_ss))
                raise arcpy.ExecuteError
            
            _sample_count = min(feat_count, _input_ss)

        self.all_features = False
        if _sample_count == feat_count:
            self.all_features = True

        arc_log_output(_sample_count)
        self.sample_size = _sample_count
        return self.sample_size


    def calculate_samples(self, feature):
        _ss = self.sample_size

        all_features = []
        with arcpy.da.SearchCursor(feature, ["OID@", "SHAPE@"]) as sc:
            for row in sc:
                all_features.append([row[0], row[1]])

        if self.all_features:
            return all_features

        random_features = random.sample(all_features, _ss)
        return random_features


    def generate_output(self, sample_features, mxd, output_folder):
        df = mxd.activeDataFrame

        if self.scale is not None:
            df.scale = self.scale

        arc_log_output(len(sample_features))
        arc_log_output(output_folder)
        arc_log_output(""  if self.scale is None else self.scale)

        _tc = len(sample_features)
        ic = int(_tc*0.01)
        if _tc < 100:
            ic = 1
        ic = max(1, ic)

        arcpy.SetProgressor("step", "Exporting Feature", 0, _tc)


        for i, feat in enumerate(sample_features):
            if (i % ic) == 0:
                arc_log_output("{} of {}".format(i+1, _tc))
                arcpy.SetProgressorPosition(i+1)

            _id, _shp = feat

            extent = _shp.extent

            if self.scale is not None:
                df.panToExtent(extent)
            else:
                df.extent = extent
                df.scale += 500

            # fl_name = "Sample_{}.tiff".format(_id)
            # fl_path = os.path.join(output_folder, fl_name)
            # arcpy.mapping.ExportToTIFF(
            #     mxd, fl_path, df,
            #     df_export_width=600, df_export_height=600
            # )

            fl_name = "Sample_{}.jpeg".format(_id)
            fl_path = os.path.join(output_folder, fl_name)
            arcpy.mapping.ExportToTIFF(
                mxd, fl_path, df,
                df_export_width=600, df_export_height=600
            )
        
        arcpy.ResetProgressor()
        
        pass


class NodeCheck(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Segmentation Validator"
        self.description = "Check and located potential segmentation errors in linear dataset"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
            displayName="Polyline Feature",
            name="polyline_feature",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )
        param0.filter.list = ["Polyline"]
        
        param1 = arcpy.Parameter(
            displayName="Output Geodatabase",
            name="out_gdb",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input",
        )
        param1.filter.list = ["Local Database"]
        
        param2 = arcpy.Parameter(
            displayName="Join Distance",
            name="join_distance",
            datatype="GPLinearUnit",
            parameterType="Required",
            direction="Input",
        )
        param2.value = "0.01 METERS"

        param3 = arcpy.Parameter(
            displayName = "Temporary Output GDB",
            name="temp_output",
            datatype="DEWorkspace",
            parameterType="Optional",
            direction="Input",
        )
        param3.filter.list = ["Local Database"]

        # Output features
        param4 = arcpy.Parameter(
            displayName = "Nodes to Check",
            name="nodes_to_check",
            datatype="GPLayer",
            parameterType="Derived",
            direction="Output"
        )

        param5 = arcpy.Parameter(
            displayName = "Nodes to Split",
            name="nodes_to_split",
            datatype="GPLayer",
            parameterType="Derived",
            direction="Output"
        )

        param6 = arcpy.Parameter(
            displayName = "Nodes to Merge",
            name="nodes_to_merge",
            datatype="GPLayer",
            parameterType="Derived",
            direction="Output"
        )
        
        return [
            param0, param1, param2, param3,
            param4, param5, param6
        ]

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
        # [arcpy.AddMessage(type(x.valueAsText)) for x in parameters]
        # return
        arcpy.env.overwriteOutput = True
        _input_polyline = parameters[0].valueAsText
        _out_folder = parameters[1].valueAsText
        _join_distance = parameters[2].valueAsText

        _temp_folder = "in_memory"
        if parameters[3].valueAsText is not None:
            _temp_folder = parameters[3].valueAsText

        nc, ns, nm = self.process_nodes(_out_folder, _temp_folder, _input_polyline, _join_distance)

        for i in [[nc, "Check", 4], [ns, "Split", 5], [nm, "Merge", 6]]:
            if i[0] is None:
                continue
            arc_log_output(i[0])
            _f_text = "Nodes to {}".format(i[1])
            _fl = arcpy.MakeFeatureLayer_management(i[0], _f_text)
            arcpy.SetParameterAsText(i[2], _f_text)
        
        return

    # Code from script

    def process_nodes(self, out_folder, temp_folder, polyline_feature, join_distance):
        arc_log_output("starting")
        verts = self.verticies(polyline_feature)
        vert_feature =  self.vertices_to_feature(verts, temp_folder)
        points = self.collect_points(vert_feature, temp_folder)
        cands = self.get_candidates(
            points, polyline_feature, temp_folder, join_distance, out_folder)
        res = self.split_results(cands, temp_folder, join_distance, out_folder)
        
        arc_log_output("end")
        return res


    def vertices_to_feature(self, vts, temp_folder):
        """Generate a feature class of supplied vertices"""
        arc_log_output("Converting vertices to feature")
        ft = arcpy.CreateFeatureclass_management(temp_folder, "verts", "POINT", spatial_reference=arcpy.SpatialReference(26917))[0]
        arcpy.AddField_management(ft, "ID", "TEXT")
        with arcpy.da.InsertCursor(ft, ["ID", "SHAPE@"]) as ic:
            for item in vts:
                ic.insertRow((item[0], item[1],))
                ic.insertRow((item[0], item[2],))
        return ft


    def get_candidates(self, verts, pipes, temp_folder, join_distance, out_folder):
        arc_log_output("Getting Locations to Check")
        sj = arcpy.SpatialJoin_analysis(verts, pipes,
                                   os.path.join(temp_folder, "join_events"),
                                   "JOIN_ONE_TO_ONE", "KEEP_ALL", "", "INTERSECT",
                                   join_distance)[0]
        eee = arcpy.Select_analysis(sj,
                                    os.path.join(out_folder, "NodesToCheck"),
                                    "\"JOIN_COUNT\" > 1")[0]
        return eee


    def split_results(self, nodes_to_check, temp_folder, join_distance, out_folder):
        arc_log_output("Getting Final Features")
        im_ft = arcpy.CopyFeatures_management(nodes_to_check,
                                      os.path.join("in_memory/itg"))[0]

        arcpy.Integrate_management(im_ft, join_distance)

        nc = None
        ns = None
        nm = None

        if int(arcpy.GetCount_management(im_ft)[0]) < 4:
            nc = arcpy.CopyFeatures_management(im_ft, os.path.join(out_folder, "NodesToCheck"))[0]
        
        else:
            ev = arcpy.CollectEvents_stats(im_ft, os.path.join(temp_folder, "events_2"))[0]
            ns = arcpy.Select_analysis(ev, os.path.join(out_folder, "NodesToSplit"), "\"ICOUNT\" = 1")[0]
            nm = arcpy.Select_analysis(ev, os.path.join(out_folder, "NodesToMerge"), "\"ICOUNT\" > 1")[0]

        return [nc, ns, nm]


    def collect_points(self, verts, temp_folder):
        """Get locations where only one vertices is present"""
        arc_log_output("Collecting Points")
        ev = arcpy.CollectEvents_stats(verts, os.path.join(temp_folder, "events_1"))[0]
        s_ev = arcpy.Select_analysis(ev, os.path.join(temp_folder, "cand_events"), "\"ICOUNT\" = 1")[0]
        return s_ev


    def verticies(self, feature):
        arc_log_output("Getting Verticies")
        out_feats = []
        errors = []
        sr = arcpy.SpatialReference(26917)
        with arcpy.da.SearchCursor(feature, ["OID@", "SHAPE@"]) as sc:
            for row in sc:
                try:
                    cent = row[1]
                    fp = arcpy.PointGeometry(cent.firstPoint, sr)
                    lp = arcpy.PointGeometry(cent.lastPoint, sr)

                    out_feats.append([row[0], fp, lp])
                except:
                    arc_log_output("Problem with feature {}".format(row[0]), "ERROR")
                # out_feats.append([row[0], cent.firstPoint, cent.lastPoint])

        return out_feats




