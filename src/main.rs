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
mod pov;
mod terraform;
mod types;
mod video;
mod world;

use std::{
    mem,
    time::{Duration, Instant}, collections::HashSet, process, sync::Arc,
};

use arrayvec::ArrayVec;
use asset::Pack;
use event::{Action, Handler, Input};
use glam::{IVec3, Quat, Vec3, EulerRot};
use pov::Pov;
use terraform::Terraformer;
use types::{Direction, SIDES, DirMap, SideMap};
use video::{
    render::{self, Renderer},
    Gfx,
};
use winit::{
    dpi::PhysicalSize,
    event_loop::{ControlFlow, EventLoop},
    keyboard::KeyCode,
    window::{CursorGrabMode, Fullscreen, WindowBuilder}, event::MouseButton,
};
use world::World;

const MAX_DISTANCE: usize = 12;
const TICK_DURATION: Duration = Duration::from_micros(31_250);

#[rustfmt::skip]
const BINDINGS: [(Input, Action); 25] = [
    (Input::Motion,                      Action::Turn),
    (Input::Close,                       Action::Exit),
    (Input::Press(KeyCode::Escape),      Action::Exit),
    (Input::Press(KeyCode::KeyQ),        Action::Exit),
    (Input::Press(KeyCode::KeyF),        Action::Debug("toggle wireframe")),
    (Input::Press(KeyCode::KeyV),        Action::Debug("toggle vsync")),
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
    let mut mesh = vec![];
    let mut alpha_mesh = vec![];
    let mut generating = HashSet::new();
    let mut pladec_queue = ArrayVec::<bool, 16>::new();
    let mut selected_block = 1;

    let mut reached_face = None;

    let _ = event_loop.run(move |event, _| {
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

            Action::Debug("change block") => {
                selected_block += event_handler.scroll_delta.1 as i16;
                selected_block = selected_block.rem_euclid(10);
            }

            _ => {}
        }

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

        let max_distance = MAX_DISTANCE as i32;

        let aim2 = Quat::from_euler(EulerRot::ZXY, -pov.yaw, -pov.pitch, 0.) * Vec3::Y;

        // jmi2k: this takes too long
        while accrued_time >= TICK_DURATION {
            #[rustfmt::skip]
            for k in z - max_distance .. z + max_distance {
            for j in y - max_distance .. y + max_distance {
            for i in x - max_distance .. x + max_distance {
                let chunk_loc = IVec3::new(i, j, k);

                let fine_loc = chunk::merge_loc(chunk_loc, IVec3::ZERO);
                let distance = pov.position.distance(fine_loc.as_vec3());
                let max_distance = MAX_DISTANCE as f32 * 32.;

                // jmi2k: this eats ~10ms at MAX_DISTANCE = 24, maybe optimize conditions? caching?
                if distance <= max_distance && world.chunk_mut(chunk_loc).is_none() && !generating.contains(&chunk_loc) {
                    terraformer.terraform(chunk_loc);
                    generating.insert(chunk_loc);
                }
            }
            }
            }

            // jmi2k: moar crap
            pov.position += aim * if sprint { 4e-1 } else { 2e-1 };
            // jmi2k: this is prohibitively expensive at MAX_DISTANCE = 24
            let time = world.tick();

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

            if time % 32 == 0 {
                eprintln!("position: {:?}   reach: {:?}", pov.position.as_ivec3(), reached_face);
            }

            accrued_time -= TICK_DURATION;
        }

        let (mut x, mut y, mut z) = pov.position.as_ivec3().into();
        let (sx, sy, sz) = aim2.signum().as_ivec3().into();
        let (dx, dy, dz) = (IVec3::new(sx, sy, sz).as_vec3() / aim2).into();
        let i = (IVec3::new(x, y, z) + IVec3::new(sx, sy, sz)).as_vec3() - pov.position;
        let (mut tx, mut ty, mut tz) = (i / aim2).into();
        let mut block = 0;
        let mut direction = Direction::Up;

        loop {
            if tx < ty {
                if tx < tz {
                    x += sx;
                    tx += dx;
                    direction = if aim2.x < 0. { Direction::East } else { Direction::West };
                } else {
                    z += sz;
                    tz += dz;
                    direction = if aim2.z < 0. { Direction::Up } else { Direction::Down };
                }
            } else {
                if ty < tz {
                    y += sy;
                    ty += dy;
                    direction = if aim2.y < 0. { Direction::North } else { Direction::South };
                } else {
                    z += sz;
                    tz += dz;
                    direction = if aim2.z < 0. { Direction::Up } else { Direction::Down };
                }
            }

            if (Vec3::new(tx, ty, tz) * aim2).length() > 10. {
                break;
            }

            let (chunk_loc, block_loc) = chunk::split_loc(IVec3 { x, y, z });
            let Some(chunk) = world.chunk(chunk_loc) else { break; };
            block = chunk[block_loc];

            if ![0, 6, 7].contains(&block) {
                break;
            }
        }

        reached_face = if ![0, 6, 7].contains(&block) { Some((IVec3 { x, y, z }, direction)) } else { None };

        if !redraw {
            return;
        }

        // jmi2k: crap crap crap

        let models = [ 
            None,
            pack.model("grass"),
            pack.model("dirt"),
            pack.model("stone"),
            pack.model("wheat_0"),
            pack.model("wheat"),
            pack.model("water"),
            pack.model("water_surface"),
            pack.model("glass"),
            pack.model("sand"),
            pack.model("wood"),
            pack.model("leaves"),
        ];

        let then = Instant::now();
        let IVec3 { x, y, z } = chunk_loc;

        #[rustfmt::skip]
        'mesh:
        for k in (z - max_distance .. z + max_distance).rev() {
        for j in y - max_distance .. y + max_distance {
        for i in x - max_distance .. x + max_distance {
            let chunk_loc = IVec3::new(i, j, k);
            let fine_loc = chunk::merge_loc(chunk_loc, IVec3::ZERO);
            let distance = pov.position.distance(fine_loc.as_vec3());
            let max_distance = MAX_DISTANCE as f32 * 32.;

            // Early continue if there is no chunk
            let Some(chunk) = world.chunk(chunk_loc) else {
                continue;
            };

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

            let neighbor_chunks = DirMap {
                west: world.chunk(chunk_loc - IVec3::X),
                east: world.chunk(chunk_loc + IVec3::X),
                south: world.chunk(chunk_loc - IVec3::Y),
                north: world.chunk(chunk_loc + IVec3::Y),
                down: world.chunk(chunk_loc - IVec3::Z),
                up: world.chunk(chunk_loc + IVec3::Z),
            };

            let neighborhood_nonces = SideMap {
                west: neighbor_chunks.west.map(|chunk| chunk.nonces().east),
                east: neighbor_chunks.east.map(|chunk| chunk.nonces().west),
                south: neighbor_chunks.south.map(|chunk| chunk.nonces().north),
                north: neighbor_chunks.north.map(|chunk| chunk.nonces().south),
                down: neighbor_chunks.down.map(|chunk| chunk.nonces().up),
                up: neighbor_chunks.up.map(|chunk| chunk.nonces().down),
                none: Some(chunk.nonces().none),
            };

            // Discard already meshed chunks immediately
            if renderer.has_mesh(chunk_loc, &neighborhood_nonces) { continue; }

            // Discard enclosed chunks immediately
            // jmi2k: can be improved, and also must take into account whether the player is inside
            if neighbor_chunks.west.map(|chunk| chunk.num_blocks() == 32_768).unwrap_or(false)
                && neighbor_chunks.east.map(|chunk| chunk.num_blocks() == 32_768).unwrap_or(false)
                && neighbor_chunks.south.map(|chunk| chunk.num_blocks() == 32_768).unwrap_or(false)
                && neighbor_chunks.north.map(|chunk| chunk.num_blocks() == 32_768).unwrap_or(false)
                && neighbor_chunks.down.map(|chunk| chunk.num_blocks() == 32_768).unwrap_or(false)
                && neighbor_chunks.up.map(|chunk| chunk.num_blocks() == 32_768).unwrap_or(false)
            {
                renderer.load_mesh(&gfx, chunk_loc, &neighborhood_nonces, &[], &[]);
                continue;
            }

            mesh.clear();
            alpha_mesh.clear();

            for side in SIDES {
                #[rustfmt::skip]
                let (Δlayer, Δrow, Δblock) = match side {
                    Some(Direction::West) =>  (IVec3::X, IVec3::Y, IVec3::Z),
                    Some(Direction::East) =>  (IVec3::X, IVec3::Z, IVec3::Y),
                    Some(Direction::South) => (IVec3::Y, IVec3::Z, IVec3::X),
                    Some(Direction::North) => (IVec3::Y, IVec3::X, IVec3::Z),
                    Some(Direction::Down) =>  (IVec3::Z, IVec3::X, IVec3::Y),
                    Some(Direction::Up) =>    (IVec3::Z, IVec3::Y, IVec3::X),
                    None =>                   (IVec3::Z, IVec3::Y, IVec3::X),
                };

                for a in 0..32 {
                for b in 0..32 {
                    let mut last_face = None;
                for c in 0..32 {
                    // Traverse chunk following the current side's canonical order
                    let block_loc = a * Δlayer + b * Δrow + c * Δblock;
                    let block = chunk[block_loc];

                    // Skip invisible blocks
                    let Some(model) = models[block as usize] else {
                        last_face = None;
                        continue;
                    };

                    // Skip empty faces
                    let Some(range) = model.ranges[side].as_ref() else {
                        last_face = None;
                        continue;
                    };

                    let translucent = (6..=7).contains(&block);

                    // jmi2k: ugly...
                    let neighbor = 'here: {
                        let Some(direction) = side else {
                            break 'here 0;
                        };

                        let neighbor_loc = block_loc + IVec3::from(direction);
                        let block_loc = chunk::mask_block_loc(neighbor_loc);

                        match (neighbor_loc.min_element(), neighbor_loc.max_element()) {
                            (-1, _) | (_, 32) => neighbor_chunks[direction].map(|chunk| chunk[block_loc]).unwrap_or(0),
                            _ => chunk[block_loc],
                        }
                    };

                    let culled = side.is_some() && (
                        (1..=3).contains(&neighbor)
                        || ((6..=7).contains(&block) && (6..=7).contains(&neighbor))
                        || (9..=11).contains(&neighbor)
                    );

                    // Skip hidden faces
                    if culled {
                        last_face = None;
                        continue;
                    }

                    for idx in range.clone() {
                        let sky_exposure = 15;
                        let mesh = if translucent { &mut alpha_mesh } else { &mut mesh };

                        // Simple inline 1D greedy meshing.
                        //
                        // This code will fail to optimize sides with more than 1 quad,
                        // but this is an acceptable limitation
                        // as those should not be optimized anyway.
                        //
                        // The face extension is done at the reference level.
                        // This is possible because the faces are canonicalized.
                        if last_face == Some((idx, sky_exposure)) {
                            let quad_ref = mesh.last_mut().unwrap();
                            render::extend_quad_ref(quad_ref, Δblock);
                        } else {
                            let quad_ref = render::quad_ref(idx, block_loc, translucent, sky_exposure);
                            mesh.push(quad_ref);
                            last_face = Some((idx, sky_exposure));
                        }
                    }
                }
                }
                }
            }

            renderer.load_mesh(&gfx, chunk_loc, &neighborhood_nonces, &mesh, &alpha_mesh);

            if then.elapsed() > Duration::from_micros(1500) {
                break 'mesh;
            }
        }
        }
        }

        let target = pov.position + aim * if sprint { 4e-1 } else { 2e-1 };
        let pov_interpolated = Pov {
            position: Vec3::lerp(pov.position, target, accrued_time.as_secs_f32() / TICK_DURATION.as_secs_f32()),
            ..pov
        };

        renderer.render(&gfx, /*&world,*/ &pov_interpolated, reached_face, MAX_DISTANCE, wireframe);
        redraw = false;
    });
}

fn open_packs() -> (Pack, Pack) {
    let pack = asset::open("packs/exthard");
    let alt_pack = asset::open("packs/nostalgia");
    // let alt_pack = asset::open("packs/exthard");

    (pack, alt_pack)
}
