from os import sys
sys.path.insert(0, '../src/')
from core import *
from pgen import *
from runners import ExperimentRunner
import time

replicates = 10
model_file = '../models/vt-coord-embeddable.mod'
base_output_dir = './data'


base_params = {
    'n_workers':100,
    'n_projects':20,
    'n_role_families':3,
    'n_role_levels':3,
    'n_roles_per_project':(2, 6),
    'horizon_project_count_ratio':1.2,
    'project_duration_range':(2, 5),
    'revenue_multiple_range':(1.5, 3),
    'rejection_penalty_multiple_range':(0.1, 0.4),
    'role_weekly_hour_range':(5, 40),
    'base_hourly_pay_range':(10, 200),
    'worker_weekly_hour_range':(5, 40),
    'assignment_multiple_range':(1, 1),
    'reassignment_multiple_range':(0.2, 0.4),
    'underutilization_hourly_cost_range':(0, 0),
    'conflicting_project_fraction':0,
    'interproject_conflicting_roles':0,
    'replaced_workers_fraction':0.1,
    'replaced_projects_fraction':0,
    'time_limit':1800
}

sp = {
    'n_workers':300,
    'n_projects':60,
    'n_role_families':5,
}

dp = {
    'horizon_project_count_ratio':0.5,
    'worker_weekly_hour_range':(20, 40),
    'conflicting_project_fraction':0.1
}

vp = {
    'replaced_workers_fraction':0.2,
    'replaced_projects_fraction':0.05
}

up = {'underutilization_hourly_cost_range':(0.1, 0.1)}

er = ExperimentRunner(base_params, sp, dp, vp, up, replicates, model_file, base_output_dir)
er.run()