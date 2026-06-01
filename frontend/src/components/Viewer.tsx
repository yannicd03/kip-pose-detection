import { Component, Suspense, useMemo, type ReactNode } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera, useGLTF } from "@react-three/drei";
import * as THREE from "three";
import type { Intrinsics, PointCloud } from "../api";
import {
  COLORS,
  T_WORLD2CVCAM,
  centroidThree,
  makeF,
  poseToMatrix,
  restingPlaneThree,
  type VizObject,
} from "../geometry";

export interface Show3D {
  grid: boolean;
  pointcloud: boolean;
  table: boolean;
  arm: boolean;
  predMesh: boolean;
  predAxes: boolean;
  gtMesh: boolean;
  gtAxes: boolean;
}

/**
 * Static scene props exported in USD world coordinates by
 * tools/usd_scene_to_glb.py. Rendered opaque (real occluders, unlike the
 * translucent pose meshes) in muted colours so they give spatial context
 * without competing with the GT/prediction overlays.
 */
const PROPS: Record<
  "table" | "arm",
  { file: string; material: THREE.MeshStandardMaterialParameters }
> = {
  table: { file: "table.glb", material: { color: "#6b7280", metalness: 0.1, roughness: 0.85 } },
  arm: { file: "arm.glb", material: { color: "#8aa1c4", metalness: 0.35, roughness: 0.5 } },
};

/**
 * Swallows load errors from an optional prop GLB (e.g. the asset was never
 * exported into public/meshes/) so a missing file degrades to "nothing drawn"
 * instead of crashing the whole canvas.
 */
class PropBoundary extends Component<
  { children: ReactNode; fallback?: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? this.props.fallback ?? null : this.props.children;
  }
}

function StaticProp({
  file,
  material,
}: {
  file: string;
  material: THREE.MeshStandardMaterialParameters;
}) {
  const { scene } = useGLTF(`/meshes/${file}`);
  const cloned = useMemo(() => {
    const c = scene.clone(true);
    // The exported GLBs carry no materials; assign one solid material and make
    // sure normals exist (quadric decimation can drop them) so lighting works.
    const mat = new THREE.MeshStandardMaterial({ side: THREE.DoubleSide, ...material });
    c.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.isMesh) {
        if (!m.geometry.getAttribute("normal")) m.geometry.computeVertexNormals();
        m.material = mat;
      }
    });
    return c;
  }, [scene, material]);
  // Geometry is in USD world coords; place it by the world->cam extrinsic via
  // the same F·T path the parts use.
  const matrix = useMemo(() => poseToMatrix(T_WORLD2CVCAM), []);
  return (
    <group matrixAutoUpdate={false} matrix={matrix}>
      <primitive object={cloned} />
    </group>
  );
}

function SceneProp({ which }: { which: "table" | "arm" }) {
  const { file, material } = PROPS[which];
  return (
    <PropBoundary>
      <Suspense fallback={null}>
        <StaticProp file={file} material={material} />
      </Suspense>
    </PropBoundary>
  );
}

function SemiMesh({
  obj,
  color,
  showMesh,
  showAxes,
}: {
  obj: VizObject;
  color: string;
  showMesh: boolean;
  showAxes: boolean;
}) {
  const { scene } = useGLTF(`/meshes/${obj.class}.glb`);
  // Clone and replace every material with a translucent solid in the layer
  // colour (semi-transparent solid, like the reference). depthWrite off so
  // overlapping GT/pred meshes blend instead of z-fighting.
  const cloned = useMemo(() => {
    const c = scene.clone(true);
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(color),
      transparent: true,
      opacity: 0.5,
      depthWrite: false,
      side: THREE.DoubleSide,
      metalness: 0.0,
      roughness: 0.65,
    });
    c.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.isMesh) m.material = mat;
    });
    return c;
  }, [scene, color]);

  const matrix = useMemo(() => poseToMatrix(obj.T), [obj.T]);

  return (
    <group matrixAutoUpdate={false} matrix={matrix}>
      <primitive object={cloned} visible={showMesh} />
      <axesHelper args={[0.04]} visible={showAxes} />
    </group>
  );
}

function PointCloudPoints({ pc }: { pc: PointCloud }) {
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const n = pc.xyz.length;
    const positions = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);
    const F = makeF();
    const v = new THREE.Vector3();
    for (let i = 0; i < n; i++) {
      const [x, y, z] = pc.xyz[i];
      v.set(x, y, z).applyMatrix4(F);
      positions[i * 3] = v.x;
      positions[i * 3 + 1] = v.y;
      positions[i * 3 + 2] = v.z;
      const c = pc.rgb[i];
      colors[i * 3] = c[0] / 255;
      colors[i * 3 + 1] = c[1] / 255;
      colors[i * 3 + 2] = c[2] / 255;
    }
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    return g;
  }, [pc]);

  return (
    <points geometry={geometry}>
      <pointsMaterial size={0.004} vertexColors sizeAttenuation />
    </points>
  );
}

/** USD world up (+Z) expressed in three.js space, via the F·T_world2cvcam path. */
function tableUp(): THREE.Vector3 {
  return new THREE.Vector3(0, 0, 1)
    .transformDirection(poseToMatrix(T_WORLD2CVCAM))
    .normalize();
}

/** A reference grid lying in the plane whose normal is `up`, centred at `position`. */
function AlignedGrid({
  position,
  up,
}: {
  position: THREE.Vector3;
  up: THREE.Vector3;
}) {
  const quaternion = useMemo(
    () =>
      new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), up),
    [up]
  );
  return (
    <gridHelper
      args={[1.5, 30, "#4a4f5a", "#2b2f38"]}
      position={position}
      quaternion={quaternion}
    />
  );
}

/**
 * Reference grid for the surface the parts rest on: oriented to the real
 * table-up axis (USD world +Z through F·T) and placed at the parts' resting
 * plane (restingPlaneThree). Deriving the height from the parts — not the
 * table mesh's tallest vertex — keeps it on the work surface instead of
 * floating up on overhead workstation structure baked into table.glb.
 */
function GroundGrid({ objects }: { objects: VizObject[] }) {
  const up = useMemo(() => tableUp(), []);
  const position = useMemo(() => restingPlaneThree(objects, up), [objects, up]);
  return <AlignedGrid position={position} up={up} />;
}

function Scene({
  W,
  H,
  K,
  objects,
  show,
  hidden,
  pointcloud,
}: {
  W: number;
  H: number;
  K: Intrinsics;
  objects: VizObject[];
  show: Show3D;
  hidden: Set<string>;
  pointcloud: PointCloud | null;
}) {
  const fovY = (2 * Math.atan((0.5 * H) / K.fy) * 180) / Math.PI;
  const aspect = W / H;
  const c = useMemo(() => centroidThree(objects), [objects]);
  // 3/4 view offset from the scene centroid so the depth is obvious at a glance.
  const camPos = useMemo<[number, number, number]>(
    () => [c.x + 0.32, c.y - 0.3, c.z + 0.55],
    [c]
  );

  return (
    <>
      <PerspectiveCamera
        makeDefault
        fov={fovY}
        aspect={aspect}
        near={0.01}
        far={100}
        position={camPos}
      />
      <ambientLight intensity={0.85} />
      <directionalLight position={[1, 2, 3]} intensity={1.1} />
      <directionalLight position={[-2, -1, 1]} intensity={0.5} />

      {show.grid && <GroundGrid objects={objects} />}

      {show.table && <SceneProp which="table" />}
      {show.arm && <SceneProp which="arm" />}

      {objects.map((o) => {
        if (hidden.has(o.key)) return null;
        const showMesh = o.kind === "gt" ? show.gtMesh : show.predMesh;
        const showAxes = o.kind === "gt" ? show.gtAxes : show.predAxes;
        return (
          <Suspense key={o.key} fallback={null}>
            <SemiMesh
              obj={o}
              color={o.kind === "gt" ? COLORS.gt : COLORS.pred}
              showMesh={showMesh}
              showAxes={showAxes}
            />
          </Suspense>
        );
      })}

      {show.pointcloud && pointcloud && <PointCloudPoints pc={pointcloud} />}

      <OrbitControls makeDefault target={[c.x, c.y, c.z]} />
    </>
  );
}

export default function Viewer3D({
  W,
  H,
  K,
  objects,
  show,
  hidden,
  pointcloud,
}: {
  W: number;
  H: number;
  K: Intrinsics;
  objects: VizObject[];
  show: Show3D;
  hidden: Set<string>;
  pointcloud: PointCloud | null;
}) {
  return (
    <div className="viewer">
      <div className="canvas-wrap canvas3d">
        <Canvas>
          <color attach="background" args={["#161922"]} />
          <Scene
            W={W}
            H={H}
            K={K}
            objects={objects}
            show={show}
            hidden={hidden}
            pointcloud={pointcloud}
          />
        </Canvas>
      </div>
    </div>
  );
}

useGLTF.preload("/meshes/anker_kurz.glb");
useGLTF.preload("/meshes/anker_lang.glb");
useGLTF.preload("/meshes/table.glb");
