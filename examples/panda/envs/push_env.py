from abc import ABC
import numpy as np

from pydrake.all import RigidTransform

from envs.panda_env import PandaEnv


class PushEnv(PandaEnv, ABC):
    """
    Dynamic pushing environment in Drake
    """
    def __init__(self, 
                dt=0.002,
                renders=False,
                visualize_contact=False,
                camera_params=None,
                hand_type='plate',
                diff_ik_filter_hz=500,
                contact_solver='sap',
                panda_joint_damping=200,
                ):
        super(PushEnv, self).__init__(
            dt=dt,
            renders=renders,
            visualize_contact=visualize_contact,
            camera_params=camera_params,
            hand_type=hand_type,
            diff_ik_filter_hz=diff_ik_filter_hz,
            contact_solver=contact_solver,
            panda_joint_damping=panda_joint_damping,
        )
        self.finger_init_pos = 0.06
        self.bottle_initial_pos = [0.45, 0.0, 0.05]

        # Set default task
        self.task = {}
        self.task['obj_mass'] = 0.2
        self.task['obj_mu'] = 0.1
        self.task['obj_modulus'] = 7
        self.task['obj_com_x'] = 0.0
        self.task['obj_com_y'] = 0.0


    def reset_task(self, task):
        return NotImplementedError


    def load_objects(self, ):
        # Load veggies - more like templates - save body and frame ids
        bottle_model_index, bottle_body_indice = \
            self.station.AddModelFromFile(
                '/examples/panda/data/bottle.sdf',
                RigidTransform([0.5, 0, 0.04]),
                name='bottle',
            )
        self.bottle_body = self.plant.get_body(bottle_body_indice[0])
        self.bottle_default_inertia = self.bottle_body.default_spatial_inertia() 


    def reset(self, task=None):
        """
        Call parent to reset arm and gripper positions (build if first-time). Reset veggies and task. Do not initialize simulator.
        """
        task = super().reset(task)

        # Get new context
        context = self.simulator.get_mutable_context()
        plant_context = self.plant.GetMyContextFromRoot(context)
        sg_context = self.sg.GetMyMutableContextFromRoot(context)
        query_object = self.sg.get_query_output_port().Eval(sg_context)
        context_inspector = query_object.inspector()

        # Set table properties
        self.set_obj_dynamics(context_inspector, 
                              sg_context, 
                              self.table_body,
                              hc_dissipation=1.0,
                              sap_dissipation=0.1,
                              mu=0.1,
                              hydro_modulus=7,
                              hydro_resolution=0.1, # does not matter
                              compliance_type='rigid')

        # Set hand properties
        self.set_obj_dynamics(context_inspector, 
                              sg_context, 
                              self.hand_body,
                              hc_dissipation=1.0,
                              sap_dissipation=0.1,
                              mu=0.3,
                              hydro_modulus=6,
                              hydro_resolution=0.1, # does not matter
                              compliance_type='rigid')

        # Set bottle properties
        self.set_obj_dynamics(context_inspector, 
                              sg_context, 
                              self.bottle_body,
                              hc_dissipation=1.0,
                              sap_dissipation=0.1,
                              mu=task['obj_mu'],
                              hydro_modulus=task['obj_modulus'],
                              hydro_resolution=0.002,  # matters
                              compliance_type='compliant')

        # First, revert inertia back to origin
        self.bottle_body.SetSpatialInertiaInBodyFrame(plant_context, self.bottle_default_inertia)

        # Next, shift the current inertia to new COM - need to shift in the opposite direction.
        inertia = self.bottle_body.CalcSpatialInertiaInBodyFrame(plant_context)
        inertia = inertia.Shift([-task['obj_com_x'], -task['obj_com_y'], 0])
        self.bottle_body.SetSpatialInertiaInBodyFrame(plant_context, inertia)

        # Reset bottle
        self.set_body_pose(self.bottle_body, plant_context, 
                            p=self.bottle_initial_pos,
                            rpy=[0, 0, 0])
        self.set_body_vel(self.bottle_body, plant_context)

        # Reset bottle
        self.set_body_pose(self.bottle_body, plant_context, 
                            p=self.bottle_initial_pos,
                            rpy=[0, 0, 0])
        self.set_body_vel(self.bottle_body, plant_context)
        station_context = self.station.GetMyContextFromRoot(context)
        return self._get_obs(station_context)


    def _get_obs(self, station_context):
        if not self.flag_use_camera:
            return None
        color_image = self.color_image_port.Eval(station_context).data[:,:,:3] # HxWx4
        color_image = np.transpose(color_image, [2,0,1])
        depth_image = np.squeeze(self.depth_image_port.Eval(station_context).data)
        
        # Normalize
        depth_image = ((self.camera_params['max_depth']-depth_image)/(self.camera_params['max_depth']-self.camera_params['min_depth']))*255
        depth_image = np.uint8(depth_image)[np.newaxis]
        image = np.vstack((color_image, depth_image))
        return image


    def visualize(self):
        raise NotImplementedError


    def get_bottle_vel(self, plant_context):
        return self.plant.EvalBodySpatialVelocityInWorld(
            plant_context, 
            self.bottle_body,
            ).get_coeffs().reshape(6,1)