import numpy as np
from typing import Tuple
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, CircularLane, AbstractLane
from highway_env.road.road import Road, LaneIndex
from highway_env.types import Vector
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle


class DemolitionDerbyEnv(AbstractEnv):
    """
    A demolition derby environment.

    Two vehicles are inclined to collide into the side of the other and
    avoid being struck on the side ("T-Boning").

    """

    CRASH_REWARD: float = 0.1
    """The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""

    GOT_CRASHED_REWARD: float = 0.4
    """The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"]."""

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "features": ['x', 'y', 'vx', 'vy', 'heading'],
                "vehicles_count": 2,
                "features_range": {
                    "x": [-100., 100.],
                    "y": [-100., 100.],
                    "vx": [-40., 40.],
                    "vy": [-40., 40.]
                    },
                "see_behind": True,
                "clip": False

            },
            "action": {
                "type": "ContinuousAction",
            },
            "screen_width": 1000,  # [px]
            "screen_height": 1000,  # [px]
            "controlled_vehicles": 1,
            "duration": 100.,  # [s]
            "derby_radius": 100.,
            "did_crash_rewards": [1.0, 1.0],
            "got_crashed_rewards": [1.0, 1.0]
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

#    def _create_road(self) -> None:
#        """ circular road that acts as boundarys for derby """
#        center = [0, 0]  # [m]
#        alpha = 24  # [deg]
#        radius = self.config["derby_radius"]
#
#        net = RoadNetwork()
#        radii = [radius, radius+4]
#        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
#        line = [[c, s], [n, c]]
#        net.add_lane("e", "w", CircularLane(center, radius, 0, np.pi, clockwise=False, line_types=line[0], speed_limit=1000))
#        net.add_lane("w", "e", CircularLane(center, radius, np.pi, 0, clockwise=False, line_types=line[0], speed_limit=1000))
#        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
#        self.road = road

    def _create_road(self) -> None:

        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = [0, 0]  # [m]

        radius = self.config["derby_radius"]  # [m]
        alpha = 90  # [deg]

        net = RoadNetwork()
        radii = [radius]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.CONTINUOUS
        line = [[c, s], [n, c]]
        for lane in [0]:

            net.add_lane("se", "ee",
                         CircularLane(center, radii[lane], np.deg2rad(-90-alpha), np.deg2rad(alpha+5),
                                      clockwise=True, line_types=line[lane]))
            net.add_lane("ee", "se",
                         CircularLane(center, radii[lane], np.deg2rad(alpha-5), np.deg2rad(92+alpha),
                                      clockwise=True, line_types=line[lane]))



        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        path = "highway_env.vehicle.derby.DerbyCar"
        vehicle_class = utils.class_from_path(path)
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            XPos = self.np_random.rand()*self.config["derby_radius"]*1.5-self.config["derby_radius"]*.75
            YPos = self.np_random.rand()*self.config["derby_radius"]*1.5-self.config["derby_radius"]*.75
            Heading = 2*np.pi*self.np_random.rand()
            Speed = 0.
            vehicle = self.action_type.vehicle_class(road=self.road, position=np.array([XPos, YPos]), heading=Heading, speed=Speed)
            vehicle = vehicle_class.create_from(vehicle)
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)


    @staticmethod
    def corner_positions(v: "Vehicle" = None)->np.array:
        """
        This method computes the position of each corner with a rotated car.

        """
        if v is None:
            print(v)
            return np.array([[0,0],[0,0],[0,0],[0,0]])
        p = v.position
        l = v.LENGTH
        w = v.WIDTH
        h = v.heading

        c, s = np.cos(h), np.sin(h)
        r = np.array([[c, -s], [s, c]])
        corners = np.array([[l*0.5,w*0.5],[-l*0.5,w*0.5],[-l*0.5,-w*0.5],[l*0.5,-w*0.5]])

        for i in range(4):
            corners[i,:]=r.dot(corners[i,:])+p

        return corners


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Step Forward in time
        Check for exiting boundary and collision
        If exit boundary, set radial velocity to zero and fix position to radius.
        """
        r = self.config["derby_radius"];
        #for iCar in range(self.config["controlled_vehicles"]):
        #    vehicle = self.road.vehicles[iCar]
        #    vehicle.act(action[iCar,:])
        collision_with_boundary=[False,False]
        # action is none as the steps above already fill the actions of the cars
        obs, reward, terminal, info = super().step(action)
       
        # checking if exits boundary
        for vehicle_index,vehicle in enumerate(self.road.vehicles):
            corners = self.corner_positions(vehicle)
            corners_r2 = np.sum(np.multiply(corners, corners),axis=1)
            max_r2 = np.max(corners_r2)
            # if a corner is beyond the circle, fix position and velocity
            if max_r2 > r*r:
                collision_with_boundary[vehicle_index]=True
                max_r = np.sqrt(max_r2)
                dr = max_r-r
                indx = np.argmax(corners_r2)
                corner = corners[indx,:]
                unitC = corner/max_r
                vel = np.array(vehicle.velocity)
                # position, movedin the direction of the unit vector of corner and magnitude dr
                vehicle.position = np.array(vehicle.position-unitC*dr)

                vehicle.speed = 0.
                ##NOT NEEDED for KINEMATICS
                # projection of velocity onto corner to center vector then setting radial velocity to zero
                #print(unitC.shape,vel.shape)
                #radial_v = np.dot(unitC, vel)
                #vel = vel - unitC*radial_v
                #vehicle.speed = np.linalg.norm(vel)
                #vehicle.direction = vel/vehicle.speed

        info["agents_rewards"] = self._agent_rewards(collision_with_boundary,action, self.controlled_vehicles)
        return [obs,reward,terminal,info]

    def _reward(self, action: np.ndarray) -> float:
        """
        Reward for hitting, and cost for being hit. +-Sin(heading difference)
        """
        return 0

    # def _agent_rewards(self, action: int, vehicles: tuple) -> float:
    #     rewards = []
    #     Speed = np.linalg.norm(self.road.vehicles[0].velocity-self.road.vehicles[1].velocity)
    #     for i, vehicle in enumerate(vehicles):
    #         reward = 0
    #         reward = self.config["did_crash_rewards"][i] * vehicle.did_crash * abs(np.sin(vehicle.crash_angle)) * (vehicle.crash_speed2)
    #         reward -= self.config["got_crashed_rewards"][i] * vehicle.got_crashed * abs(np.sin(vehicle.crash_angle)) * (vehicle.crash_speed2)
    #         rewards.append(reward)
    #
    #     return tuple(rewards)
    def _agent_rewards(self, collision_with_boundary:np.ndarray,action: int, vehicles: tuple) -> float:
            rewards = []
            distance = pow((vehicles[0].position[0] - vehicles[1].position[0])**2 + (vehicles[0].position[1] - vehicles[1].position[1])**2,0.5)
            distance_inv = 100 * np.exp(-distance) - 1#pow(10,-distance) - 1#100 * np.exp(-distance) - 1# - np.abs(action[0][1])# - np.abs(action[0][1]);
            Speed = np.linalg.norm(self.road.vehicles[0].velocity-self.road.vehicles[1].velocity)
            for i, vehicle in enumerate(vehicles):
                
                reward = distance_inv
                # print(distance_inv)
                reward += self.config["did_crash_rewards"][i] * vehicle.did_crash * abs(np.sin(vehicle.crash_angle)) * (vehicle.crash_speed2)
                reward -= self.config["got_crashed_rewards"][i] * vehicle.got_crashed * abs(np.sin(vehicle.crash_angle)) * (vehicle.crash_speed2)
                if(collision_with_boundary[i]):
                     reward=reward-100
                rewards.append(reward)
                # print(reward)
            return tuple(rewards)

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] * self.config["simulation_frequency"]
            #(self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)

class MultiAgentDemolitionDerbyEnv(DemolitionDerbyEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                 "type": "MultiAgentAction",
                 "action_config": {
                     "type": "ContinuousAction",
                 }
            },
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "features": ['x', 'y', 'vx', 'vy', 'heading'],
                    "vehicles_count": 2,
                    "features_range": {
                        "x": [-100., 100.],
                        "y": [-100., 100.],
                        "vx": [-40., 40.],
                        "vy": [-40., 40.]
                        },
                    "see_behind": True,
                    "clip": False
                }
            },
            "controlled_vehicles": 2,
            "duration": 100.,  # [s]
            "derby_radius": 100.,
            "did_crash_rewards": [1.0, 1.0],
            "got_crashed_rewards": [1.0, 1.0]
        })
        return config


TupleMultiAgentDemolitionDerbyEnv = MultiAgentWrapper(MultiAgentDemolitionDerbyEnv)

register(
    id='demolition_derby-v0',
    entry_point='highway_env.envs:DemolitionDerbyEnv',
)
register(
    id='demolition_derby-multi-agent-v0',
    entry_point='highway_env.envs:MultiAgentDemolitionDerbyEnv',
)

register(
    id='demolition_derby-multi-agent-v1',
    entry_point='highway_env.envs:TupleMultiAgentDemolitionDerbyEnv',
)
