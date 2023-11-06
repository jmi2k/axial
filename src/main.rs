#![feature(const_trait_impl)]
#![feature(extend_one)]
#![feature(impl_trait_in_assoc_type)]
#![feature(isqrt)]
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
    time::{Duration, Instant}, collections::HashSet,
};

use asset::Pack;
use event::{Action, Handler, Input};
use glam::{IVec3, Quat, Vec3};
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
    window::{CursorGrabMode, Fullscreen, WindowBuilder},
};
use world::World;

const TICK_DURATION: Duration = Duration::from_micros(31_250);

#[rustfmt::skip]
const BINDINGS: [(Input, Action); 19] = [
    (Input::Motion,                      Action::Turn),
    (Input::Close,                       Action::Exit),
    (Input::Press(KeyCode::Escape),      Action::Exit),
    (Input::Press(KeyCode::KeyQ),        Action::Exit),
    (Input::Press(KeyCode::KeyE),        Action::Debug("reload packs")),
    (Input::Press(KeyCode::Backquote),   Action::Debug("switch packs")),
    (Input::Press(KeyCode::Tab),         Action::Fullscreen),
    (Input::Press(KeyCode::KeyW),        Action::Walk(Direction::North)),
    (Input::Press(KeyCode::KeyA),        Action::Walk(Direction::West)),
    (Input::Press(KeyCode::KeyS),        Action::Walk(Direction::South)),
    (Input::Press(KeyCode::KeyD),        Action::Walk(Direction::East)),
    (Input::Press(KeyCode::Space),       Action::Walk(Direction::Up)),
    (Input::Press(KeyCode::ShiftLeft),   Action::Walk(Direction::Down)),
    (Input::Release(KeyCode::KeyW),      Action::Stop(Direction::North)),
    (Input::Release(KeyCode::KeyA),      Action::Stop(Direction::West)),
    (Input::Release(KeyCode::KeyS),      Action::Stop(Direction::South)),
    (Input::Release(KeyCode::KeyD),      Action::Stop(Direction::East)),
    (Input::Release(KeyCode::Space),     Action::Stop(Direction::Up)),
    (Input::Release(KeyCode::ShiftLeft), Action::Stop(Direction::Down)),
];

#[pollster::main]
async fn main() {
    let (mut pack, mut alt_pack) = open_packs();
    let mut event_handler = Handler::from(BINDINGS);
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = WindowBuilder::new()
        .with_title("aXial")
        .with_inner_size(PhysicalSize::new(854, 480))
        .build(&event_loop)
        .unwrap();

    let mut gfx = Gfx::new(&window).await;
    window.set_cursor_grab(CursorGrabMode::Confined).unwrap();
    window.set_cursor_visible(false);

    let mut renderer = Renderer::new(&gfx, &pack);
    let mut world = World::new();
    let mut pov = Pov::default();

    let mut redraw = false;
    let mut then = Instant::now();
    let mut accrued_time = Duration::ZERO;
    let mut walks = DirMap::default();

    // jmi2k: crap
    #[allow(clippy::unusual_byte_groupings, reason = "digits form words")]
    let terraformer = Terraformer::new(0xA11_1DEA5_FA11_DEAD);
    let mut mesh = vec![];
    let mut runtime = Duration::ZERO;
    let mut frames = 0;
    let mut generating = HashSet::new();
    let distance = 8;

    let _ = event_loop.run(move |event, target| {
        match event_handler.handle(event) {
            Action::Exit => {
                eprintln!("{} fps", 1_000_000 * frames / runtime.as_micros());
                target.exit();
            }

            Action::Fullscreen if window.fullscreen().is_none() => {
                let mode = Fullscreen::Borderless(None);
                window.set_fullscreen(Some(mode));
            }

            Action::Fullscreen if window.fullscreen().is_some() => {
                window.set_fullscreen(None);
            }

            Action::Turn => {
                let (x, y) = event_handler.delta;
                pov.yaw += x as f32 * 1e-3;
                pov.pitch += y as f32 * 1e-3;
            }

            Action::Walk(direction) => {
                walks[direction] = true;
            }

            Action::Stop(direction) => {
                walks[direction] = false;
            }

            Action::Redraw => {
                redraw = true;
            }

            Action::Resize(new_size) => {
                gfx.resize_viewport(new_size);
            }

            Action::Debug("reload packs") => {
                (pack, alt_pack) = open_packs();
                renderer.update_pack(&gfx, &pack);
            }

            Action::Debug("switch packs") => {
                mem::swap(&mut pack, &mut alt_pack);
                renderer.update_pack(&gfx, &pack);
            }

            _ => {}
        }

        let now = Instant::now();
        accrued_time += now - then;
        runtime += now - then;
        then = now;

        // jmi2k: crap
        let aim = Quat::from_rotation_z(-pov.yaw) * IVec3::from(&walks).as_vec3();
        let (chunk_loc, _) = chunk::split_loc(pov.position.as_ivec3());
        let IVec3 { x, y, z } = chunk_loc;

        if let Ok((location, chunk)) = terraformer.chunk_rx().try_recv() {
            world.load(location, chunk);
            generating.remove(&location);
        }

        while accrued_time >= TICK_DURATION {
            #[rustfmt::skip]
            for k in z - distance .. z + distance {
            for j in y - distance .. y + distance {
            for i in x - distance .. x + distance {
                let chunk_loc = IVec3::new(i, j, k);

                if world.chunk(chunk_loc).is_none() && !generating.contains(&chunk_loc) {
                    terraformer.terraform(chunk_loc);
                    generating.insert(chunk_loc);
                }
            }
            }
            }

            // jmi2k: moar crap
            pov.position += aim * 1.;

            world.tick();
            accrued_time -= TICK_DURATION;
        }

        if !redraw {
            return;
        }

        // jmi2k: crap crap crap

        let models = [ 
            pack.model("grass").unwrap(),
            pack.model("dirt").unwrap(),
            pack.model("stone").unwrap(),
            pack.model("wheat_0").unwrap(),
            pack.model("wheat").unwrap(),
        ];

        #[rustfmt::skip]
        for k in z - distance .. z + distance {
        for j in y - distance .. y + distance {
        for i in x - distance .. x + distance {
            let chunk_loc = IVec3::new(i, j, k);

            let Some(chunk) = world.chunk(chunk_loc) else {
                continue;
            };

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

            if renderer.has_mesh(chunk_loc, &neighborhood_nonces) { continue; }
            mesh.clear();

            for z in 0..32 {
            for y in 0..32 {
            for x in 0..32 {
                let block_loc = IVec3 { x, y, z };
                let block = chunk[block_loc];

                if chunk[block_loc] == 0 {
                    continue;
                }

                let block_neighbors = DirMap {
                    west: if x == 0 { neighbor_chunks.west.map(|chunk| chunk[block_loc + 31 * IVec3::X]).unwrap_or(0) } else { chunk[block_loc - IVec3::X] },
                    east: if x == 31 { neighbor_chunks.east.map(|chunk| chunk[block_loc - 31 * IVec3::X]).unwrap_or(0) } else { chunk[block_loc + IVec3::X] },
                    south: if y == 0 { neighbor_chunks.south.map(|chunk| chunk[block_loc + 31 * IVec3::Y]).unwrap_or(0) } else { chunk[block_loc - IVec3::Y] },
                    north: if y == 31 { neighbor_chunks.north.map(|chunk| chunk[block_loc - 31 * IVec3::Y]).unwrap_or(0) } else { chunk[block_loc + IVec3::Y] },
                    down: if z == 0 { neighbor_chunks.down.map(|chunk| chunk[block_loc + 31 * IVec3::Z]).unwrap_or(0) } else { chunk[block_loc - IVec3::Z] },
                    up: if z == 31 { neighbor_chunks.up.map(|chunk| chunk[block_loc - 31 * IVec3::Z]).unwrap_or(0) } else { chunk[block_loc + IVec3::Z] },
                };

                for side in SIDES {
                    if side.map(|direction| (1..=3).contains(&block_neighbors[direction])).unwrap_or(false) {
                        continue;
                    }

                    let range = models[block as usize - 1].ranges[side].as_ref();

                    if range.is_none() {
                        continue;
                    }

                    for idx in range.unwrap().clone() {
                        let sky_exposure = 0xF;
                        let quad_ref = render::quad_ref(idx, block_loc, sky_exposure);
                        mesh.push(quad_ref);
                    }
                }
            }
            }
            }

            renderer.load_mesh(&gfx, chunk_loc, &neighborhood_nonces, &mesh);
        }
        }
        }

        let target = pov.position + aim * 1.;
        let pov_interpolated = Pov {
            position: Vec3::lerp(pov.position, target, accrued_time.as_secs_f32() / TICK_DURATION.as_secs_f32()),
            ..pov
        };

        renderer.render(&gfx, &world, &pov_interpolated);
        frames += 1;
        redraw = false;
    });
}

fn open_packs() -> (Pack, Pack) {
    let pack = asset::open("packs/exthard");
    //let alt_pack = asset::open("packs/nostalgia");
    let alt_pack = asset::open("packs/exthard");

    (pack, alt_pack)
}
