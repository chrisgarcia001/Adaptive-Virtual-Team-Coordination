import os
import random as rnd
from functools import reduce
from core import *
from pgen import *
import util as ut



# This builds a 2^k factorial param combinations set with corresponding combination names. Names are in the form of
# 'mppm' which for example means minus-plus-plus-minus with respect to the factor levels.
# Example usage: 
#    build_factorial_param_combs({0:0}, [{1:1}, {2:2}]) ==> [('mm', {0: 0}), ('mp', {0: 0, 2: 2}), ('pm', {0: 0, 1: 1}), ('pp', {0: 0, 1: 1, 2: 2})]
# @param base_params: a dict representing params at ALL LOW level
# @high_levels: a list of dicts, where each dict represents the HIGH level for a particular factor
# @returns: a list in form [(<combination name>, <params>)
def build_factorial_param_combs(base_params, high_levels):
    merge = lambda d1, d2: dict(list(d1.items()) + list(d2.items()))
    if len(high_levels) == 1:
        return [('m', dict(base_params)), ('p', merge(base_params, high_levels[0]))]
    else:
        low = [('m' + x, y) for (x, y) in build_factorial_param_combs(base_params, high_levels[1:])]
        high = [('p' + x, y) for (x, y) in build_factorial_param_combs(merge(base_params, high_levels[0]), high_levels[1:])]
        return low + high


class ExperimentRunner:
    def __init__(self, base_params, sp, dp, vp, up, replicates, model_file, base_output_dir):
        self.base_params = base_params
        self.sp = sp
        self.dp = dp
        self.vp = vp
        self. up = up
        self.replicates = replicates
        self.base_output_dir = base_output_dir
        self.base_hourly_range = range(base_params['base_hourly_pay_range'][0], base_params['base_hourly_pay_range'][1] + 1)
        self.model = model_file
               
    
    # rs = RoleSkillGenerator([[10,11,12], [30,34,36], [27,34]], worker_num_skill_histogram=[0.8, 0.2])
    # Builds a random RoleSkillGenerator. 
    #    role_levels_pay input generated to randomly-generated hourly wages drawn from self.base_hourly_range, put in ascending order for each level.
    #    worker_num_skill_histogram input generated as each role family having equal number of skills.
    def build_role_skill_generator(self, n_role_families, n_role_levels, role_weekly_hour_range, hour_increment):
        n_part = lambda x, n: [sorted(x[:n])] + n_part(x[n:], n) if len(x) > n else [sorted(x)]  # n_part(list(range(9)), 3) => [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        role_levels_pay = n_part(rnd.sample(list(self.base_hourly_range), n_role_families * n_role_levels), n_role_levels)
        worker_num_skill_histogram = [1.0 / n_role_families for i in range(n_role_families)]
        return RoleSkillGenerator(role_levels_pay, weekly_hour_range=role_weekly_hour_range, 
                                  worker_num_skill_histogram=worker_num_skill_histogram, hour_increment=hour_increment)
    
    # This method runs a single problem (initial and updated) based on the specified parameters.
    def execute_problem(self, params, problem_name, replicate_num):
        p = params
        pn = str(problem_name) + '_' + str(replicate_num)
        n_conflicting_projects = int(p['n_projects'] * p['conflicting_project_fraction'])
        rs = self.build_role_skill_generator(p['n_role_families'], p['n_role_levels'], p['role_weekly_hour_range'], hour_increment=p['hour_increment'])
        pg1 = ProblemGenerator(p['n_workers'], 
                               p['n_projects'], 
                               p['n_roles_per_project'], 
                               rs, 
                               int(p['horizon_project_count_ratio'] * p['n_projects']), 
                               p['project_duration_range'],
                               p['revenue_multiple_range'], 
                               p['rejection_penalty_multiple_range'], 
                               p['worker_weekly_hour_range'], 
                               p['assignment_multiple_range'], 
                               p['reassignment_multiple_range'], 
                               p['underutilization_hourly_cost_range'], 
                               n_conflicting_projects, 
                               p['interproject_conflicting_roles'],
                               p['hour_increment'])
        pg1.build_new_problem()
        print('Solving Initial Problem:', pn, '...')
        pg1.problem.to_opl_data(os.path.join(self.base_output_dir, 'initial_problem_data', pn + '.dat'))
        solver = OplrunSolver(self.model)
        init_result = solver.solve(pg1.problem, time_limit=p['time_limit'], 
                                   solution_filename=os.path.join(self.base_output_dir, 'initial_solution_data', pn + '.txt'))
        print('Done!')
        print('Solving Updated Problem:', pn, '...')
        n_replaced_workers = int(p['replaced_workers_fraction'] * p['n_workers'])
        n_replaced_projects = int(p['replaced_projects_fraction'] * p['n_projects'])
        pg1.delete(n_replaced_workers, n_replaced_projects)
        pg1.add(n_replaced_workers, n_replaced_projects)
        pg1.problem.to_opl_data(os.path.join(self.base_output_dir, 'updated_problem_data', pn + '.dat'))
        updated_result = solver.solve(pg1.problem, time_limit=p['time_limit'], 
                                      solution_filename=os.path.join(self.base_output_dir, 'updated_solution_data', pn + '.txt'))
        print('Done!')
        return {'name':pn, 'inital_solution':init_result, 'updated_solution':updated_result}
    
    # This method runs the experiment across all parameter combinations and replicates
    def run(self):
        summary_output = [['Problem', 'InitialGap', 'InitialTime', 'UpdatedGap', 'UpdatedTime']]
        named_param_combs = build_factorial_param_combs(self.base_params, [self.sp, self.dp, self.vp, self.up])
        for (prob_class, params) in named_param_combs:
            for i in range(1, self.replicates + 1):
                result = self.execute_problem(params, prob_class, i)
                name, ins, ups = result['name'], result['inital_solution'], result['updated_solution']
                summary_output.append([name, ins['gap'], ins['time'], ups['gap'], ups['time']])
        print('Writing summary.csv ...')
        summary_text = "\n".join([','.join([str(x) for x in y]) for y in summary_output])
        ut.write_file(summary_text, os.path.join(self.base_output_dir, 'summary.csv'))
        print('Done!')
                
        