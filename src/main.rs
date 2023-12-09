#![feature(extend_one)]
#![feature(iter_collect_into)]
#![feature(lint_reasons)]
#![feature(new_uninit)]
#![feature(slice_flatten)]
#![feature(stmt_expr_attributes)]
#![feature(variant_count)]

mod asset;
mod chunk;
mod event;
mod mesh;
mod pov;
mod terraform;
mod types;
mod video;
mod world;

use std::{
    mem,
    time::{Duration, Instant}, collections::HashSet, process, sync::Arc, ops::Neg,
};

use arrayvec::ArrayVec;
use asset::Pack;
use chunk::Chunk;
use event::{Action, Handler, Input};
use glam::{IVec3, Quat, Vec3, EulerRot, IVec2};
use mesh::Mesher;
use pov::Pov;
use terraform::Terraformer;
use types::{Direction, SIDES, DirMap, SideMap};
use video::{
    render::{self, Renderer, QuadRef},
    Gfx,
};
use winit::{
    dpi::PhysicalSize,
    event_loop::{ControlFlow, EventLoop},
    keyboard::KeyCode,
    window::{CursorGrabMode, Fullscreen, WindowBuilder}, event::MouseButton,
};
use world::World;

const MAX_RENDER_DISTANCE: usize = 12;
const MAX_TICK_DISTANCE: usize = 4;
const MAX_REACH: f32 = 10.;
const TICK_DURATION: Duration = Duration::from_micros(31_250);

#[rustfmt::skip]
const BINDINGS: [(Input, Action); 27] = [
    (Input::Motion,                      Action::Turn),
    (Input::Close,                       Action::Exit),
    (Input::Press(KeyCode::Escape),      Action::Exit),
    (Input::Press(KeyCode::KeyQ),        Action::Exit),
    (Input::Press(KeyCode::KeyF),        Action::Debug("toggle wireframe")),
    (Input::Press(KeyCode::KeyV),        Action::Debug("toggle vsync")),
    (Input::Press(KeyCode::KeyG),        Action::Debug("toggle greedy meshing")),
    (Input::Press(KeyCode::KeyE),        Action::Debug("reload packs")),
    (Input::Press(KeyCode::Backquote),   Action::Debug("switch packs")),
    (Input::Press(KeyCode::Tab),         Action::Fullscreen),
    (Input::Press(KeyCode::ControlLeft), Action::Sprint),
    (Input::Press(KeyCode::KeyW),        Action::Walk(Direction::North)),
    (Input::Press(KeyCode::KeyA),        Action::Walk(Direction::West)),
    (Input::Press(KeyCode::KeyS),        Action::Walk(Direction::South)),
    (Input::Press(KeyCode::KeyD),        Action::Walk(Direction::East)),
    (Input::Press(KeyCode::Space),       Action::Walk(Direction::Up)),
    (Input::Press(KeyCode::ShiftLeft),   Action::Walk(Direction::Down)),
    (Input::Unpress(KeyCode::KeyW),      Action::Stop(Direction::North)),
    (Input::Unpress(KeyCode::KeyA),      Action::Stop(Direction::West)),
    (Input::Unpress(KeyCode::KeyS),      Action::Stop(Direction::South)),
    (Input::Unpress(KeyCode::KeyD),      Action::Stop(Direction::East)),
    (Input::Unpress(KeyCode::Space),     Action::Stop(Direction::Up)),
    (Input::Unpress(KeyCode::ShiftLeft), Action::Stop(Direction::Down)),
    (Input::Click(MouseButton::Left),    Action::Debug("destroy block")),
    (Input::Click(MouseButton::Right),   Action::Debug("place block")),
    (Input::Click(MouseButton::Middle),  Action::Debug("clone block")),
    (Input::Scroll,                      Action::Debug("change block")),
];

#[pollster::main]
async fn main() {
    let (mut pack, mut alt_pack) = open_packs();
    let mut event_handler = Handler::from(BINDINGS);
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = Arc::new(WindowBuilder::new()
        .with_title("aXial")
        .with_inner_size(PhysicalSize::new(854, 480))
        .build(&event_loop)
        .unwrap());

    let mut pov = Pov::default();
    let mut walks = DirMap::default();
    let mut sprint = false;

    let mut gfx = Gfx::new(window.clone()).await;
    window.set_cursor_grab(CursorGrabMode::Confined).unwrap();
    window.set_cursor_visible(false);

    let mut renderer = Renderer::new(&gfx, &pack);
    let mut redraw = false;
    let mut wireframe = false;
    let mut world = World::new();

    let mut then = Instant::now();
    let mut accrued_time = Duration::ZERO;

    // jmi2k: crap
    #[allow(clippy::unusual_byte_groupings, reason = "digits form words")]
    let terraformer = Terraformer::new(0xA11_1DEA5_FA11_DEAD);
    let mut mesher = Mesher::new();
    let mut generating = HashSet::new();
    let mut pladec_queue = ArrayVec::<bool, 16>::new();
    let mut selected_block = 1;

    let mut reached_face = None;
    let mut offset_3d = IVec3::ZERO;

    let _ = event_loop.run(move |event, _| {
        // Process user input
        match event_handler.handle(event) {
            Action::Exit => {
                process::exit(0);
            }

            Action::Fullscreen if window.fullscreen().is_none() => {
                let mode = Fullscreen::Borderless(None);
                window.set_fullscreen(Some(mode));
            }

            Action::Fullscreen if window.fullscreen().is_some() => {
                window.set_fullscreen(None);
            }

            Action::Turn => {
                let (x, y) = event_handler.cursor_delta;
                pov.yaw += x as f32 * 1e-3;
                pov.pitch += y as f32 * 1e-3;
            }

            Action::Walk(direction) => {
                walks[direction] = true;
            }

            Action::Stop(direction) => {
                walks[direction] = false;
                sprint &= IVec3::from(&walks) != IVec3::ZERO;
            }

            Action::Sprint => {
                sprint |= IVec3::from(&walks) != IVec3::ZERO;
            }

            Action::Redraw => {
                redraw = true;
            }

            Action::Resize(new_size) => {
                gfx.resize_viewport(new_size);
            }

            Action::Debug("toggle wireframe") => {
                wireframe = !wireframe;
            }

            Action::Debug("toggle vsync") => {
                gfx.toggle_vsync();
            }

            Action::Debug("toggle greedy meshing") => {
                mesher.greedy_meshing = (mesher.greedy_meshing + 1) % 3;
                renderer.invalidate_meshes();
            }

            Action::Debug("reload packs") => {
                (pack, alt_pack) = open_packs();
                renderer.update_pack(&gfx, &pack);
            }

            Action::Debug("switch packs") => {
                mem::swap(&mut pack, &mut alt_pack);
                renderer.update_pack(&gfx, &pack);
            }

            Action::Debug("place block") => {
                pladec_queue.push(true);
            }

            Action::Debug("destroy block") => {
                pladec_queue.push(false);
            }

            Action::Debug("clone block") => {
                if let Some((location, _)) = reached_face {
                    selected_block = world.block(location).unwrap_or_default();
                }
            }

            Action::Debug("change block") => {
                selected_block += event_handler.scroll_delta.1 as i16;
                selected_block = selected_block.rem_euclid(10);
            }

            _ => {}
        }

        // Collect time
        let now = Instant::now();
        accrued_time += now - then;
        then = now;

        // jmi2k: crap
        let aim = Quat::from_rotation_z(-pov.yaw) * IVec3::from(&walks).as_vec3();
        let (chunk_loc, _) = chunk::split_loc(pov.position.as_ivec3());
        let IVec3 { x, y, z } = chunk_loc;

        while let Ok((location, chunk, _)) = terraformer.chunk_rx().try_recv() {
            world.load(location, chunk);
            generating.remove(&location);
        }

        // jmi2k: this takes too long
        while accrued_time >= TICK_DURATION {
            let max_distance = MAX_RENDER_DISTANCE as i32;

            #[rustfmt::skip]
            for k in z - max_distance .. z + max_distance {
            for j in y - max_distance .. y + max_distance {
            for i in x - max_distance .. x + max_distance {
                let chunk_loc = IVec3::new(i, j, k);

                let fine_loc = chunk::merge_loc(chunk_loc, IVec3::ZERO);
                let distance = pov.position.distance(fine_loc.as_vec3());
                let max_distance = MAX_RENDER_DISTANCE as f32 * 32.;

                // jmi2k: this eats ~10ms at MAX_DISTANCE = 24, maybe optimize conditions? caching?
                if distance <= max_distance && world.chunk_mut(chunk_loc).is_none() && !generating.contains(&chunk_loc) {
                    terraformer.terraform(chunk_loc);
                    generating.insert(chunk_loc);
                }
            }
            }
            }

            // jmi2k: moar crap
            pov.position += aim * if sprint { 4e-0 } else { 2e-1 };
            // jmi2k: this is prohibitively expensive at MAX_DISTANCE = 24
            world.tick(pov.position);

            if let Some((location, direction)) = reached_face {
                //let mut world = world::<1>::new(&mut world);
                for action in &pladec_queue {
                    if *action {
                        world.place(location + IVec3::from(direction), selected_block);
                    } else {
                        world.destroy(location);
                    }
                }
            }

            pladec_queue.clear();
            accrued_time -= TICK_DURATION;
        }

        reached_face = cast_ray(&mut world, &pov);

        if !redraw {
            return;
        }

        let then = Instant::now();
        let mut num_checked = 0;

        while num_checked < 8 * MAX_RENDER_DISTANCE * MAX_RENDER_DISTANCE * MAX_RENDER_DISTANCE {
            let max_distance = MAX_RENDER_DISTANCE as i32;

            num_checked += 1;
            offset_3d.x += 1;

            if offset_3d.x >= 2 * max_distance {
                offset_3d.x = 0;
                offset_3d.y += 1;
            }

            if offset_3d.y >= 2 * max_distance {
                offset_3d.y = 0;
                offset_3d.z += 1;
            }

            if offset_3d.z >= 2 * max_distance {
                offset_3d.z = 0;
            }

            let chunk_loc = IVec3 { x, y, z } - max_distance + offset_3d;
            let fine_loc = chunk::merge_loc(chunk_loc, IVec3::ZERO);
            let distance = pov.position.distance(fine_loc.as_vec3());
            let max_distance = MAX_RENDER_DISTANCE as f32 * 32.;

            // Unload far away chunks
            if distance > max_distance {
                continue;
            }

            // Early continue if there is no chunk
            let Some(chunk) = world.chunk(chunk_loc) else {
                continue;
            };

            // Discard empty chunks immediately
            if chunk.num_blocks() == 0 {
                continue;
            }

            let neighbor_chunks = SideMap::from_fn(|side| {
                let Δ = side.map(IVec3::from).unwrap_or(IVec3::ZERO);
                world.chunk(chunk_loc + Δ)
            });

            let neighbor_nonces = SideMap::from_fn(|side| {
                let opposite_side = side.map(Neg::neg);
                neighbor_chunks[side].map(|chunk| chunk.nonces()[opposite_side])
            });

            // Discard already meshed chunks immediately
            if renderer.has_mesh(chunk_loc, &neighbor_nonces) { continue; }

            let (mesh, alpha_mesh) = mesher.mesh(&pack, &neighbor_chunks);
            renderer.load_mesh(&gfx, chunk_loc, &neighbor_nonces, mesh, alpha_mesh);

            if then.elapsed() > Duration::from_micros(3500) {
                break;
            }
        }

        // Interpolate POV
        let target = pov.position + aim * if sprint { 4e-0 } else { 2e-1 };
        let ratio = accrued_time.as_secs_f32() / TICK_DURATION.as_secs_f32();
        let pov = pov.lerp(target, ratio);

        // Render next frame
        renderer.render(&gfx, /*&world,*/ &pov, reached_face, MAX_RENDER_DISTANCE, wireframe);
        redraw = false;
    });
}

fn open_packs() -> (Pack, Pack) {
    let pack = asset::open("packs/exthard");
    let alt_pack = asset::open("packs/nostalgia");
    // let alt_pack = asset::open("packs/exthard");

    (pack, alt_pack)
}

fn cast_ray(world: &mut World, pov: &Pov) -> Option<(IVec3, Direction)> {
    let mut location = pov.position.as_ivec3();
    let orientation = Quat::from_euler(EulerRot::ZXY, -pov.yaw, -pov.pitch, 0.);
    let forward = orientation * Vec3::Y;
    let step = forward.signum().as_ivec3();
    let Δ = forward.signum() / forward;
    let foo = location.as_vec3() + step.max(IVec3::ZERO).as_vec3() - pov.position;
    let mut t = foo / forward;
    let mut direction = Direction::Up;

    loop {
        if t.x < t.y {
            if t.x < t.z {
                location.x += step.x;
                t.x += Δ.x;
                direction = if forward.x < 0. { Direction::East } else { Direction::West };
            } else {
                location.z += step.z;
                t.z += Δ.z;
                direction = if forward.z < 0. { Direction::Up } else { Direction::Down };
            }
        } else {
            if t.y < t.z {
                location.y += step.y;
                t.y += Δ.y;
                direction = if forward.y < 0. { Direction::North } else { Direction::South };
            } else {
                location.z += step.z;
                t.z += Δ.z;
                direction = if forward.z < 0. { Direction::Up } else { Direction::Down };
            }
        }

        if (t * forward).length() > MAX_REACH {
            return None;
        }

        let block = world.block(location)?;

        if ![0, 6, 7].contains(&block) {
            return Some((location, direction));
        }
    }
}
