
import arcpy

f = r"C:\Users\scody\Desktop\MDL Update\ALL_PIPES_20200304_GMBP\ALL_PIPES_20200304_GA.OUT\SCENARIO\AVE_DAY_CURRENT\JunctOut.dbf"

sc = arcpy.da.SearchCursor(f, "*")


z = 1