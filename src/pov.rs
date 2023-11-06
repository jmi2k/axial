use glam::{EulerRot, Mat4, Vec3};

use std::f32::consts::FRAC_PI_2;

#[derive(Default)]
pub struct Pov {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
}

impl From<&Pov> for Mat4 {
    fn from(pov: &Pov) -> Self {
        let z_up = Mat4::from_rotation_x(-FRAC_PI_2);
        let rotation = Mat4::from_euler(EulerRot::XZY, pov.pitch, pov.yaw, 0.);
        let xlation = Mat4::from_translation(-pov.position);

        z_up * rotation * xlation
    }
}
