use glam::{EulerRot, Mat4, Vec3, IVec3};

use std::f32::consts::FRAC_PI_2;

#[derive(Default)]
pub struct Pov {
    // jmi2k: separate into position and location for smooth movement far away
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
}

impl Pov {
    pub fn lerp(&self, target: Vec3, ratio: f32) -> Self {
        Self {
            position: self.position.lerp(target, ratio),
            ..*self
        }
    }

    pub fn location(&self) -> IVec3 {
        let position = -self.position;
        position.floor().as_ivec3()
    }

    pub fn xform(&self) -> Mat4 {
        let position = -self.position;
        let fract_pos = position.fract();

        let z_up = Mat4::from_rotation_x(-FRAC_PI_2);
        let rotation = Mat4::from_euler(EulerRot::XZY, self.pitch, self.yaw, 0.);
        let xlation = Mat4::from_translation(fract_pos);

        z_up * rotation * xlation
    }
}
