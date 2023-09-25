# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 18:35:49 2023

@author: u300737
"""
import cartopy.io.shapereader as shpr
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.prepared import prep

land_shp_fname = shpr.natural_earth(resolution='10m', category='physical', name='land')
land_geom = unary_union(list(shpr.Reader(land_shp_fname).geometries()))
land = prep(land_geom)

def is_land(x, y):
    return land.contains(sgeom.Point(x, y))
